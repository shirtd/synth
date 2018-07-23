import argparse

parser = {
    'make' : argparse.ArgumentParser(description='create csv files from pickle chunks.'),
    'persist' : argparse.ArgumentParser(description='generate persistent features.'),
    'main' : argparse.ArgumentParser(description='construct synthetic features.'),
    'join' : argparse.ArgumentParser(description='join csvs.'),
    'run' : argparse.ArgumentParser(description='persist, main, and join.'),
    'predict' : argparse.ArgumentParser(description='topological predict.')
}

# MAKE
parser['make'].add_argument('-p', '--src', default=None, help='source (pickle file) directory. set default in util/config.py')
parser['make'].add_argument('-d', '--dest', default='data', help='destination directory.')
parser['make'].add_argument('-c', '--chunk', type=int, default=0, help='chunk # (0-14). default: 0')
parser['make'].add_argument('-n', '--rows', type=int, default=None, help='number of rows to write for each modulation. default: 2000 (all)')
parser['make'].add_argument('-s', '--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser['make'].add_argument('--force', action='store_true', help='force reset.')

# PERSIST
# MAKE
parser['persist'].add_argument('--dir', default='data', help='input data directory. default: data')
parser['persist'].add_argument('-c', '--chunk', type=int, default=0, help='chunk # (0-14). default: 0')
parser['persist'].add_argument('-s', '--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser['persist'].add_argument('-n','--n', type=int, default=100, help='landscape length (resolution). default: 100')
parser['persist'].add_argument('-k','--k', type=int, default=10, help='number of landscape features (degree). default: 10')
parser['persist'].add_argument('--force', action='store_true', help='force reset.')

# MAIN
# parser['main'].add_argument('-c', '--chunk', type=int, default=0, help='training data chunk (0-14). default: 0')
# parser['main'].add_argument('--test', type=int, default=-1, help='test data chunk (0-14). -1 for no test. default: -1')
# parser['main'].add_argument('--snr', type=int, default=10, help='signal-to-noise ratio (+-10, +-6, +-2,). default: 10')
parser['main'].add_argument('-c', '--chunk', type=int, default=0, help='training data chunk (0-14). default: 0')
parser['main'].add_argument('--test', type=int, default=-1, help='test data chunk (0-14). -1 for no test. default: -1')
parser['main'].add_argument('-s','--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser['main'].add_argument('--suff', default='', help='suffix. default: none.')
parser['main'].add_argument('--dir', default='data', help='input data directory. default: data')
parser['main'].add_argument('--tdir', default='data_out', help='output data directory. default: data_out')
parser['main'].add_argument('--jdir', default='data_out_json', help='json data directory. default: data_out_json')

parser['main'].add_argument('--config', default='config/batch.conf', help='batch config file.')
parser['main'].add_argument('-r','--restart', action='store_true', help='restart batch decomp.')

parser['main'].add_argument('-t', '--transpose', action='store_true', help='compute and upload transpose source.')
parser['main'].add_argument('-i', '--init', action='store_true', help='initialize source.')
parser['main'].add_argument('-w', '--write', action='store_true', help='write group files.')
parser['main'].add_argument('-b', '--batch', action='store_true', help='run batch decompositon.')
parser['main'].add_argument('-j', '--json', action='store_true', help='write feature json files.')
parser['main'].add_argument('--synth', action='store_true', help='run synthetic feature generation.')
parser['main'].add_argument('--fun', default='activations', help='synthetic feature function. default: activations.')
parser['main'].add_argument('-f', '--force', action='store_true', help='force reset.')
parser['main'].add_argument('-u', '--upload', action='store_true', help='upload sources.')
parser['main'].add_argument('--trainpath', default=None, help='train synth base file name.')
parser['main'].add_argument('--testpath', default=None, help='test synth base file name.')


# JOIN

parser['join'].add_argument('-c', '--chunk', type=int, default=0, help='training data chunk (0-14). default: 0')
parser['join'].add_argument('-s','--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser['join'].add_argument('--suff', nargs='+', help='suffixes of files to add. default: none.')
# parser['join'].add_argument('-b', '--base', help='base file.')
# parser['join'].add_argument('-f', '--files', nargs='+', help='files to join.')
# parser['join'].add_argument('-o', '--out', help='file out.')
parser['join'].add_argument('--force', action='store_true', help='force reset.')
parser['join'].add_argument('-u', '--upload', action='store_true', help='upload sources.')
parser['join'].add_argument('--dir', default='data', help='input/output data directory. default: data')

# PREDICT
parser['predict'].add_argument('-T', '--train', default='data/SNR10_0_synth_ph.csv', help='training data file. default: data/SNR10_0_synth_ph.csv.')
parser['predict'].add_argument('-t', '--test', default='data/SNR10_1_synth_ph.csv', help='testing data file. default: data/SNR10_1_synth_ph.csv.')
parser['predict'].add_argument('-r', '--rows', type=int, default=None, help='number of rows to test per modulation. default: None (all).')
parser['predict'].add_argument('-g', '--gt', type=int, default=1, help='ground truth (target) column index. default: 1.')
parser['predict'].add_argument('-k', '--k', default=15, help='number of nearest neighbors. default: 10.')
parser['predict'].add_argument('-m', '--m', default=24, help='number of classes. default: 24')
parser['predict'].add_argument('-f', '--fun', default='euclidean', help='knn metric function.')
parser['predict'].add_argument('-c', '--col', default='allbut', help='column set. default: allbut.')
parser['predict'].add_argument('-n', '--net', default='allbut', help='network. default: same as column set.')

# RUN
parser['run'].add_argument('-c', '--chunk', type=int, default=0, help='chunk # (0-14). default: 0')
parser['run'].add_argument('-s', '--snr', type=int, nargs='+', default=None, help='signal-to-noise ratios (+-10, +-6, +-2,). default: None (all)')
parser['run'].add_argument('-n','--n', type=int, default=100, help='landscape length (resolution). default: 100')
parser['run'].add_argument('-k','--k', type=int, default=10, help='number of landscape features (degree). default: 10')
parser['run'].add_argument('--force', action='store_true', help='force reset.')
parser['run'].add_argument('--test', type=int, default=-1, help='test data chunk (0-14). -1 for no test. default: -1')
parser['run'].add_argument('--suff', default='', help='suffix. default: none.')
parser['run'].add_argument('--dir', default='data', help='input data directory. default: data')
parser['run'].add_argument('--tdir', default='data_out', help='output data directory. default: data_out')
parser['run'].add_argument('--jdir', default='data_out_json', help='json data directory. default: data_out_json')
parser['run'].add_argument('--config', default='config/batch.conf', help='batch config file.')
parser['run'].add_argument('-r','--restart', action='store_true', help='restart batch decomp.')
parser['run'].add_argument('-t', '--transpose', action='store_true', help='compute and upload transpose source.')
parser['run'].add_argument('-i', '--init', action='store_true', help='initialize source.')
parser['run'].add_argument('-w', '--write', action='store_true', help='write group files.')
parser['run'].add_argument('-b', '--batch', action='store_true', help='run batch decompositon.')
parser['run'].add_argument('-j', '--json', action='store_true', help='write feature json files.')
parser['run'].add_argument('--synth', action='store_true', help='run synthetic feature generation.')
parser['run'].add_argument('--fun', default='projection', help='synthetic feature function. default: activations.')
parser['run'].add_argument('-u', '--upload', action='store_true', help='upload sources.')
