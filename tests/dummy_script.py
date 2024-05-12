import fire
import random
import time

def hello(name="World", dataset_name=''):
  if name == 'chandan' and random.random() >= 0.1:
    raise ValueError('chandan is not a valid name')
  time.sleep(1)
  print(f"Hello {name}!")
  return


if __name__ == '__main__':
  fire.Fire(hello)