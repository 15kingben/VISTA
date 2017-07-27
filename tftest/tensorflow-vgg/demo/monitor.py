import os
import time
import subprocess


# os.system('source ~/tensorflow/bin/activate.csh')
count = 0
while True:
        count += 1
        if count >= 330:
                os.system('pkill "sshd\: bking11"')
                count = 0
	if os.listdir('input') == []:
		print('nothing')
		time.sleep(5)
	else:
                # os.system('rm output/output.txt')
		fail=False
		# Check validiy of input
		d = os.listdir('input')[0]
                os.system('mkdir output/' + d)
                old_size = len(os.listdir('input/'+ d ))
                ec = 0
                while ec < 10:
                        if old_size != 0:
                                break
                        ec+=1
                        time.sleep(1)
                        old_size = len(os.listdir('input/' + d))
                while True:
                        time.sleep(1)
                        new_size = len(os.listdir('input/' + d))
                        if new_size == old_size:
                                break
                        else:
                                old_size = new_size

                for i in os.listdir('input/' + d):
			if i[-4:] not in ['.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG']:
				fail=True


		if fail:
			print('cant read image file')
			os.system('rm input/' + d + '/*')			
			continue

		print('Machine Learning')
		os.chdir('../')
		try:
			x = subprocess.check_output(['/localdisk/bking11/tensorflow/bin/python', 'runbinary_vgg19_trainable.py', d])
			os.chdir('demo/')
			print(x)
			os.system('rm -r input/' + d )
		except subprocess.CalledProcessError as e:
			print(e.output)
			os.chdir('demo')
			os.system('rm -r input/' + d)
		except OSError:
			print('poop')
			os.chdir('demo')
			os.system('rm -r input/' + d )
