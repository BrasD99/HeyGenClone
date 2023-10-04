import os, glob, subprocess, argparse, sys, numpy, random, math, cv2
from itertools import repeat
from multiprocessing import Pool
from scipy.io import wavfile
from pydub import AudioSegment
from tqdm import tqdm

def get_length(input_video):
	result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	return float(result.stdout)

def read_Vox_lines(file):
	Tlines, Flines = [], []	
	with open(file) as f_in:
		while True:
			line = f_in.readline()					
			if not line:
				break
			if int(line[0]):
				Tlines.append(line)
			else:
				Flines.append(line)
	return Tlines, Flines

def read_LRS3_ST(file):
	lines = []
	with open(file) as f_in:
		while True:
			line = f_in.readline()
			if not line:
				break
			lines.append(line)
	return lines[:30000]

def read_LRS3_S(file):
	lines = []
	with open(file) as f_in:
		while True:
			line = f_in.readline()
			if not line:
				break
			start = int(line.split()[1]) / 100
			end = int(line.split()[2]) / 100
			if end - start <= 3: # Only select less than 3s
				lines.append(line)
	return lines[:30000]

def generate_TAudio(line, args):
	# Get the id of the audio and video
	audio_name = line.split()[1][:-4]
	video_name = line.split()[2][:-4]
	id1 = audio_name.split('/')[0]
	name1 = audio_name.split('/')[0] + '_' + audio_name.split('/')[1] + '_' + audio_name.split('/')[2]
	name2 = video_name.split('/')[0] + '_' + video_name.split('/')[1] + '_' + video_name.split('/')[2]
	name = name1 + '_' + name2
	audio_path = os.path.join(args.Vox_audio, audio_name + '.wav')
	video_path = os.path.join(args.Vox_video, video_name + '.mp4')
	out_audio_path = os.path.join(args.out_path, 'TAudio', id1 + '/' + name + '.wav')
	out_video_path = os.path.join(args.out_path, 'TAudio', id1 + '/' + name + '.mp4')
	os.makedirs(os.path.join(args.out_path, 'TAudio', id1), exist_ok = True)
	
	# Read the audio data and the length of audio and video
	audio = AudioSegment.from_file(audio_path, format="wav")
	length_audio = len(audio) / 1000.0
	length_video = get_length(video_path)
	length_data = int(min(length_video, length_audio) * 100) / 100
	audio = audio[:int(length_data * 1000)]	

	# Extract the video and audio
	start = 0
	end = length_data
	audio.export(out_audio_path, format="wav")	
	cmd = "ffmpeg -y -ss %.3f -t %.3f -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest %s -loglevel panic"% (start, end - start, video_path, out_audio_path, out_video_path)
	subprocess.call(cmd, shell=True, stdout=None)

	# # Write the txt file
	start_T, end_T = 0, length_data
	start_F, end_F= 0, 0
	line_new = "TAudio" +  ' ' + str(audio_name) + ' ' + str(video_name) + ' ' + str(length_data) \
	+ ' ' + str(start_T) + ' ' + str(end_T) + ' ' + str(start_F) + ' ' + str(end_F) + '\n'
	return line_new

def generate_FAudio(line, args):
	# Get the id of the audio and video
	audio_name = line.split()[1][:-4]
	video_name = line.split()[2][:-4]
	id1 = audio_name.split('/')[0]
	name1 = audio_name.split('/')[0] + '_' + audio_name.split('/')[1] + '_' + audio_name.split('/')[2]
	name2 = video_name.split('/')[0] + '_' + video_name.split('/')[1] + '_' + video_name.split('/')[2]
	name = name1 + '_' + name2
	audio_path = os.path.join(args.Vox_audio, audio_name + '.wav')
	video_path = os.path.join(args.Vox_video, video_name + '.mp4')
	out_audio_path = os.path.join(args.out_path, 'FAudio', id1 + '/' + name + '.wav')
	out_video_path = os.path.join(args.out_path, 'FAudio', id1 + '/' + name + '.mp4')
	os.makedirs(os.path.join(args.out_path, 'FAudio', id1), exist_ok = True)

	# Read the audio data and the length of audio and video	
	audio = AudioSegment.from_file(audio_path, format="wav")
	length_audio = len(audio) / 1000.0
	length_video = get_length(video_path)
	length_data = int(min(length_video, length_audio) * 100) / 100
	audio = audio[:int(length_data * 1000)]

	# Extract the video and audio
	start = 0
	end = length_data
	audio.export(out_audio_path, format="wav")	
	cmd = "ffmpeg -y -ss %.3f -t %.3f -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest %s -loglevel panic"% (start, end - start, video_path, out_audio_path, out_video_path)
	subprocess.call(cmd, shell=True, stdout=None)

	# Write the txt file
	start_T, end_T = 0, 0
	start_F, end_F= 0, length_data
	line_new = "FAudio" +  ' ' + str(audio_name) + ' ' + str(video_name) + ' ' + str(length_data) \
	+ ' ' + str(start_T) + ' ' + str(end_T) + ' ' + str(start_F) + ' ' + str(end_F) + '\n'
	return line_new

def generate_TFAudio(line, args):
	# Get the id of the audio and video
	audio_name = line.split()[1][:-4]
	video_name = line.split()[2][:-4]
	id1 = audio_name.split('/')[0]
	name1 = audio_name.split('/')[0] + '_' + audio_name.split('/')[1] + '_' + audio_name.split('/')[2]
	name2 = video_name.split('/')[0] + '_' + video_name.split('/')[1] + '_' + video_name.split('/')[2]
	name = name1 + '_' + name2
	audio_T_path = os.path.join(args.Vox_audio, video_name + '.wav')
	audio_F_path = os.path.join(args.Vox_audio, audio_name + '.wav')
	video_path = os.path.join(args.Vox_video, video_name + '.mp4')
	out_audio_path = os.path.join(args.out_path, 'TFAudio', id1 + '/' + name + '.wav')
	out_video_path = os.path.join(args.out_path, 'TFAudio', id1 + '/' + name + '.mp4')
	os.makedirs(os.path.join(args.out_path, 'TFAudio', id1), exist_ok = True)

	# Read the audio data and the length of audio and video	
	audio_T = AudioSegment.from_file(audio_T_path, format="wav")
	audio_F = AudioSegment.from_file(audio_F_path, format="wav")
	length_audio_T = len(audio_T) / 1000.0
	length_audio_F = len(audio_F) / 1000.0
	length_video = get_length(video_path)
	length_data = int(min(length_audio_T, length_audio_F, length_video) * 100) / 100
	audio_T = audio_T[:int(length_data * 1000)]
	audio_F = audio_F[:int(length_data * 1000)]

	# Generate the audio
	changepoint = int((length_data * 0.25 + length_data * random.random() * 0.5) * 100) / 100
	audio_dict = {}
	audio_dict['T1'] = audio_T[:changepoint * 1000]
	audio_dict['T2'] = audio_T[changepoint * 1000:]
	audio_dict['F1'] = audio_F[:changepoint * 1000]
	audio_dict['F2'] = audio_F[changepoint * 1000:]
	seed = random.randint(0,1)
	if seed == 1:
		audio = audio_dict['T1'] + audio_dict['F2']
	else:
		audio = audio_dict['F1'] + audio_dict['T2']	
	# Extract the video and audio
	start = 0
	end = length_data
	audio.export(out_audio_path, format="wav")	
	cmd = "ffmpeg -y -ss %.3f -t %.3f -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest %s -loglevel panic"% (start, end - start, video_path, out_audio_path, out_video_path)
	subprocess.call(cmd, shell=True, stdout=None)

	# Write the txt file
	if seed == 1:
		start_T, end_T, start_F, end_F = 0, changepoint, changepoint, length_data
	elif seed == 0:
		start_F, end_F, start_T, end_T = 0, changepoint, changepoint, length_data
	line_new = "TFAudio" +  ' ' + str(audio_name) + ' ' + str(video_name) + ' ' + str(length_data) \
	+ ' ' + str(start_T) + ' ' + str(end_T) + ' ' + str(start_F) + ' ' + str(end_F) + '\n'
	return line_new

def generate_TSilence(line, args):
	# Get the id of the audio and video
	type_change = line.split()[0]
	audio_name = line.split()[1]
	video_name = line.split()[1]
	id1 = audio_name.split('/')[0]
	name1 = audio_name.split('/')[0] + '_' + audio_name.split('/')[1] + '_' + line.split()[5]
	name2 = video_name.split('/')[0] + '_' + video_name.split('/')[1] + '_' + line.split()[5]
	name = name1 + '_' + name2
	start = int(line.split()[2]) / 100
	mid = int(line.split()[3]) / 100
	end = int(line.split()[4]) / 100
	audio_path = os.path.join(args.lrs3_audio, 'pretrain', audio_name[8:] + '.wav')
	video_path = os.path.join(args.lrs3_video, 'pretrain', video_name[8:]+ '.mp4')
	out_audio_path = os.path.join(args.out_path, 'TSilence', id1 + '/' + name + '.wav')
	out_video_path = os.path.join(args.out_path, 'TSilence', id1 + '/' + name + '.mp4')
	os.makedirs(os.path.join(os.path.join(args.out_path, 'TSilence'), id1), exist_ok = True)

	# Read the audio data and the length of audio and video	
	audio = AudioSegment.from_file(audio_path, format="wav")

	# Get the required audio and video data
	length_data = int((end - start) * 100) / 100
	audio = audio[int(start * 1000):int(end * 1000)]
	audio.export(out_audio_path, format="wav")	
	cmd = "ffmpeg -y -ss %.3f -t %.3f -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest %s -loglevel panic"% (start, end - start, video_path, out_audio_path, out_video_path)
	subprocess.call(cmd, shell=True, stdout=None)

	changepoint = int((mid - start) * 100) / 100
	if type_change == "10":
		start_T, end_T, start_F, end_F = 0, changepoint, changepoint, length_data
	elif type_change == "01":
		start_T, end_T, start_F, end_F = changepoint, length_data, 0,  changepoint

	audio_name = audio_name[:-5] + line.split()[5]
	video_name = video_name[:-5] + line.split()[5]
	line_new = "TSilence" +  ' ' + str(audio_name) + ' ' + str(video_name) + ' ' + str(length_data) \
	+ ' ' + str(start_T) + ' ' + str(end_T) + ' ' + str(start_F) + ' ' + str(end_F) + '\n'
	return line_new

def generate_FSilence(line, Flines, args):
	# Get the id of the audio and video
	audio_T_name = line.split()[0]
	video_name = line.split()[0]
	start = int(line.split()[1]) / 100
	end = int(line.split()[2]) / 100
	length_data = int((end - start) * 100) / 100	
	changepoint = int((length_data * 0.25 + length_data * random.random() * 0.5) * 100) / 100
	speech_line = random.choice(Flines)
	length_speech = float(speech_line.split()[-1])
	while length_speech < length_data:
		speech_line = random.choice(Flines)
		length_speech = float(speech_line.split()[-1])
	audio_F_name = speech_line.split()[1][:-4]
	id1 = audio_F_name.split('/')[0]
	name1 = audio_F_name.split('/')[0] + '_' + audio_F_name.split('/')[1] + '_' + audio_F_name.split('/')[2]
	name2 = audio_T_name.split('/')[0] + '_' + audio_T_name.split('/')[1] + '_' + line.split()[-1]
	name = name1 + '_' + name2

	# True: orig_video False: speech+slience
	video_path = os.path.join(args.lrs3_video, 'pretrain', video_name[8:]+ '.mp4')
	audio_T_path = os.path.join(args.lrs3_audio, 'pretrain', audio_T_name[8:] + '.wav')
	audio_F_path = os.path.join(args.Vox_audio, audio_F_name + '.wav')
	out_audio_path = os.path.join(args.out_path, 'FSilence', id1 + '/' + name + '.wav')
	out_video_path = os.path.join(args.out_path, 'FSilence', id1 + '/' + name + '.mp4')
	os.makedirs(os.path.join(args.out_path, 'FSilence', id1), exist_ok = True)

	# Read the audio data and the length of audio and video	
	audio_T = AudioSegment.from_file(audio_T_path, format="wav")
	audio_T = audio_T[int(start * 1000):int(end * 1000)]
	audio_F = AudioSegment.from_file(audio_F_path, format="wav")
	length_audio_T = len(audio_T) / 1000.0
	length_audio_F = len(audio_F) / 1000.0
	length_video = get_length(video_path)
	length_data = int(min(length_audio_T, length_audio_F, length_video) * 100) / 100
	audio_T = audio_T[:int(length_data * 1000)]
	audio_F = audio_F[:int(length_data * 1000)]

	# Generate the audio
	audio_dict = {}
	audio_dict['T1'] = audio_T[:changepoint * 1000]
	audio_dict['T2'] = audio_T[changepoint * 1000:]
	audio_dict['F1'] = audio_F[:changepoint * 1000]
	audio_dict['F2'] = audio_F[changepoint * 1000:]
	seed = random.randint(0,1)
	if seed == 1:
		audio = audio_dict['T1'] + audio_dict['F2']
	else:
		audio = audio_dict['F1'] + audio_dict['T2']	
	# Extract the video and audio
	audio.export(out_audio_path, format="wav")	
	cmd = "ffmpeg -y -ss %.3f -t %.3f -i %s -i %s -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest %s -loglevel panic"% (start, end - start, video_path, out_audio_path, out_video_path)
	subprocess.call(cmd, shell=True, stdout=None)

	# Write the txt file
	if seed == 1:
		start_T, end_T, start_F, end_F = 0, changepoint, changepoint, length_data
	elif seed == 0:
		start_F, end_F, start_T, end_T = 0, changepoint, changepoint, length_data
	
	video_name = video_name[:-5] + line.split()[-1]
	line_new = "FSilence" +  ' ' + str(audio_F_name) + ' ' + str(video_name) + ' ' + str(length_data) \
	+ ' ' + str(start_T) + ' ' + str(end_T) + ' ' + str(start_F) + ' ' + str(end_F) + '\n'
	return line_new

# MAIN
parser = argparse.ArgumentParser(description = "generate_Dataset")

parser.add_argument('--List_folder',   type=str, default= 'lists')
parser.add_argument('--out_path',   type=str, default= '/data07/ruijie/database/TalkSet_final')
parser.add_argument('--Vox_audio',   type=str, default= '/home/ruijie/database/VoxCeleb2/audio/audio_clean/clean/train')
parser.add_argument('--Vox_video',   type=str, default= '/home/ruijie/database/VoxCeleb2/video/orig/train')
parser.add_argument('--lrs3_audio', type=str, default='/data07/ruijie/database/LRS3/audio/orig_audio/clean')
parser.add_argument('--lrs3_video', type=str, default='/data07/ruijie/database/LRS3/video/orig_video')
parser.add_argument('--task',   type=str, default='TAudio')
parser.add_argument('--num_cpu',   type=int, default=10)
args = parser.parse_args()

os.makedirs(os.path.join(args.out_path, 'TAudio'), exist_ok = True)
os.makedirs(os.path.join(args.out_path, 'FAudio'), exist_ok = True)
os.makedirs(os.path.join(args.out_path, 'TFAudio'), exist_ok = True)
os.makedirs(os.path.join(args.out_path, 'FSilence'), exist_ok = True)
os.makedirs(os.path.join(args.out_path, 'TSilence'), exist_ok = True)

args.list_Vox = os.path.join(args.List_folder, 'lists_in', 'Vox_list.txt')
args.list_LRS3_S = os.path.join(args.List_folder, 'lists_in', 'LRS3_S_list.txt')
args.list_LRS3_ST = os.path.join(args.List_folder, 'lists_in', 'LRS3_ST_list.txt')
args.list_out = os.path.join(args.List_folder, 'lists_out')
args.list_out_train = os.path.join(args.list_out, 'train.txt')
args.list_out_test = os.path.join(args.list_out, 'test.txt')

if args.task == 'TAudio':
	Tlines, _ = read_Vox_lines(args.list_Vox)
	Tlines_new = []
	# Generate the video and audio
	with Pool(args.num_cpu) as p:
		Tlines_new.append(p.starmap(generate_TAudio, zip(Tlines, repeat(args))))
	# Write the txt file
	out_Tlist_file = open(os.path.join(args.list_out, 'TAudio.txt'), "w")
	for line_new in Tlines_new[0]:
		out_Tlist_file.write(line_new)
	print('TAudio Finish')

if args.task == 'FAudio':
	_, Flines = read_Vox_lines(args.list_Vox)
	Flines_new = []
	# Generate the video and audio
	with Pool(args.num_cpu) as p:
		Flines_new.append(p.starmap(generate_FAudio, zip(Flines, repeat(args))))

	# Write the txt file
	out_Flist_file = open(os.path.join(args.list_out, 'FAudio.txt'), "w")
	for line_new in Flines_new[0]:
		out_Flist_file.write(line_new)
	print('FAudio Finish')

if args.task == 'TFAudio':
	_, Flines = read_Vox_lines(args.list_Vox)
	TFlines_new = []
	# Generate the video and audio
	with Pool(args.num_cpu) as p:
		TFlines_new.append(p.starmap(generate_TFAudio, zip(Flines, repeat(args))))

	# Write the txt file
	out_TFlist_file = open(os.path.join(args.list_out, 'TFAudio.txt'), "w")
	for line_new in TFlines_new[0]:
		out_TFlist_file.write(line_new)
	print('TFAudio Finish')

if args.task == 'TSilence':	
	Slines = read_LRS3_ST(args.list_LRS3_ST)
	TSlines_new = []
	with Pool(args.num_cpu) as p:
		TSlines_new.append(p.starmap(generate_TSilence, zip(Slines, repeat(args))))
	
	# Write the txt file
	out_TSlist_file = open(os.path.join(args.list_out, 'TSilence.txt'), "w")
	for line_new in TSlines_new[0]:
		out_TSlist_file.write(line_new)
	print('TSilence Finish')

if args.task == 'FSilence':	
	Tlines, _ = read_Vox_lines(args.list_Vox)
	Slines = read_LRS3_S(args.list_LRS3_S)
	FSlines_new = []
	with Pool(args.num_cpu) as p:
		FSlines_new.append(p.starmap(generate_FSilence, zip(Slines, repeat(Tlines), repeat(args))))

	out_FSlist_file = open(os.path.join(args.list_out, 'FSilence.txt'), "w")
	for line_new in FSlines_new[0]:
		out_FSlist_file.write(line_new)
	print('FSilence Finish')

if args.task == 'Fusion':
	lines = []
	for name in {'TAudio', 'FAudio', 'TFAudio', 'TSilence', 'FSilence'}:
		with open(args.list_out + '/' + name + '.txt') as f:
			while True:
				line = f.readline()
				if not line:
					break
				lines.append(line)
	train_file = open(args.list_out_train, "w")
	test_file = open(args.list_out_test, "w")
	random.shuffle(lines)
	for num, line in enumerate(lines):
		data = line.split()
		if float(data[3]) > 6: # For the data longer than 6s, we cut them into 6s, so that will make training process simple.
			line = str(data[0]) +  ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + \
				   str(min(float(data[3]), 6)) + ' ' + str(min(float(data[4]), 6)) + ' ' + \
				   str(min(float(data[5]), 6)) + ' ' + str(min(float(data[6]), 6)) + ' ' + \
				   str(min(float(data[7]), 6)) + ' ' + "%06d"%int(num) + '\n'
		else:
			line = str(data[0]) +  ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + \
				   str(data[3]) + ' ' + str(data[4]) + ' ' + \
				   str(data[5]) + ' ' + str(data[6]) + ' ' + \
				   str(data[7]) + ' ' + "%06d"%int(num) + '\n'
		if num % 30000 < 27000:
			train_file.write(line)
		else:
			test_file.write(line)