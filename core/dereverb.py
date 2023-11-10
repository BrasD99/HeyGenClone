import os
import torch
from tqdm import tqdm
import numpy as np
import librosa
import onnxruntime as ort
import soundfile as sf
from core.temp_manager import TempFileManager


class Conv_TDF_net_trim:
    def __init__(
        self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(Conv_TDF_net_trim, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )
        self.target_name = target_name
        self.blender = 'blender' in model_name

        out_c = self.dim_c * 4 if target_name == '*' else self.dim_c
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == '*' else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])


def get_models(device, dim_f, dim_t, n_fft):
    return Conv_TDF_net_trim(
        device=device,
        model_name='Conv-TDF',
        target_name='vocals',
        L=11,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


class Predictor:
    def __init__(self, args):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)
        self.args = args
        self.model_ = get_models(
            device=self.device, dim_f=args.dim_f, dim_t=args.dim_t, n_fft=args.n_fft
        )
        self.model = ort.InferenceSession(
            os.path.join(args.onnx, self.model_.target_name + '.onnx'),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        print('onnx load done')

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.args.margin
        chunk_size = self.args.chunks * 44100
        assert not margin == 0, 'margin cannot be zero!'
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        '''
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        '''
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description('Processing')
        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros(
                    (2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i: i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(
                mix_waves, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.args.denoise:
                    spec_pred = (
                        -_ort.run(None, {'input': -
                                  spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None,
                                   {'input': spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    input_data = spek.cuda().numpy() if self.device.type == 'cuda' else spek.cpu().numpy()
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {'input': input_data})[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[
                    ::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        progress_bar.close()
        return _sources

    def prediction(self, m):
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T
        temp_manager = TempFileManager()
        voice_temp_file = temp_manager.create_temp_file(suffix='.wav')
        noise_temp_file = temp_manager.create_temp_file(suffix='.wav')
        sf.write(
            voice_temp_file, mix - opt, rate
        )
        sf.write(noise_temp_file, opt, rate)

        return {
            'voice_file': voice_temp_file.name,
            'noise_file': noise_temp_file.name
        }


class MDXNetDereverb:
    def __init__(self, chunks):
        self.onnx = os.path.join('weights', 'onnx_dereverb_By_FoxJoy')
        if not os.path.exists(self.onnx):
            raise Exception('No weights found')
        self.shifts = 10
        self.mixing = 'min_mag'
        self.chunks = chunks
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.pred = Predictor(self)

    def split(self, input):
        return self.pred.prediction(input)
