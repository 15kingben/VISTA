import os
import subprocess

instap_folders = ["beerbong/", "collegeparty/", "frathouse/", "kegstand/", "shotgunbeer/", \
	"beerfunnel/" , "flipcup/",  "kegger/" ,   "partypeople/"]
flickrp_folders = ['beerbong/' , 'beerpong/', 'chuggingalcohol/', 'chuggingbeer/', 'drunkfrat/', \
	'fratpartydrunk/', 'kegstand/' , 'passedoutdrunk/', 'shotgunbeer/', \
	'underagedrinking/', 'overfittest/']



targets = ['beerpong', 'collegeparty', 'frat', 'shotgunbeer','flipcup', 'beerfunnel', 'keg', 'passed', 'chug', 'party', 'underagedrinking']

# 'beerbong',  

for target in targets:
	if not os.path.exists("./" + target + '/'):
		os.mkdir('./' + target)
	ps = subprocess.Popen(('python', 'train_vgg19_trainable.py', target), stdout = subprocess.PIPE)
	output = subprocess.check_output(('tee', '-a','log' + '_' + target + '.txt'), stdin=ps.stdout)

