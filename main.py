
import time
import os
from random import randint
def plant():
    crop = randint(10000, 99999) # Number Id
    
    land = open('dump/outputs.txt', 'a')
    land.write(f'\n{crop}')
    land.close()

    return crop

def main(amount):
    for i in range(amount):
        planted_crop = plant() # Get new crop
        print('\n', '-'*50)
        os.system(f'git commit -am "#{planted_crop}"')
        # os.system('git push --all')
        time.sleep(0.1) # Delay to prevent read-write overlap

crop_amount = int(input('Crop Amount: '))
    
if __name__ == '__main__':
    main(crop_amount)
