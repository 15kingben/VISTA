import skimage.io
import os

base = '../../Scraping/flickr/mmfeat/images_2/'
base2 = '../../Scraping/instagram_scraper/instagram_scraper/instagram/'
ext = ['picture/', 'jpg/', 'imgnet/']
ext2 = ['friends']


for x in ext2:
	for i in os.listdir(base2 + x):
                # print(base2 + x + i)
		try:
                        skimage.io.imread(base2 + x + i)
 		except IOError:
 			print(base2 + x + i)
                        os.remove(base2 + x + i)



