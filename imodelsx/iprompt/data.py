import os
import pandas as pd

def get_add_two_dataset():
    df = pd.read_csv(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'add_two.csv'
    ), delimiter=',')
    df['output_strings'] = df['output_strings'].str.replace("'", "")
    return df['input_strings'], df['output_strings']