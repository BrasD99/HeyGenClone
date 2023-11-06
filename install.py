import subprocess

def parse_requirements(filename):
    requirements = []
    with open('requirements.txt', 'r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if line:
                requirements.append(line)
    return requirements

def install_requirements(requirements):
    for i, requirement in enumerate(requirements):
        package_name = requirement
        if '=' in requirement:
            package_name = requirement.split('=')[0]
        elif ' ' in requirement:
            package_name = '[bugs fix]'
        print(f'[{i + 1}/{len(requirements)}] Installing package "{package_name}"...')
        subprocess.run(f'pip install {requirement}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

requirements = parse_requirements('requirements.txt')
install_requirements(requirements)
print('Done!')