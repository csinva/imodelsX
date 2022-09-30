import embgam.data as data
import config
from os.path import join as oj

if __name__ == '__main__':
    rs_vary_ngrams_test = data.load_fitted_results(fname_filters=['ngtest'])
    rs_vary_ngrams_test.to_pickle(oj(config.results_dir, 'rs_vary_ngrams_test.pkl'))

    # rs = data.load_fitted_results()
    # rs.to_pickle(oj(config.results_dir, 'fitted_results_aggregated.pkl'))