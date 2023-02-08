import os
import pandas as pd

def get_add_two_numbers_dataset(num_examples: int= None):
    df = pd.read_csv(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'add_two.csv'
    ), delimiter=',')
    df['output_strings'] = df['output_strings'].str.replace("'", "")
    if num_examples is not None:
        df = df.sample(n=num_examples)
    inputs, outputs = df['input_strings'].values, [v.replace('\\n', '\n') for v in df['output_strings'].values]
    return inputs, outputs

if __name__ == '__main__':
    inputs, outputs = get_add_two_numbers_dataset()
    print(inputs[:5])
    print(outputs[:5])