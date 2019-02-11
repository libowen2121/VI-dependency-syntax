from optparse import OptionParser
from vi_syntax.vi.Session import Session


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--data_path',            dest='data_path',           metavar='FILE',
                      default='',   help='')
    parser.add_option('--train_fname',          dest='train_fname',         metavar='FILE',
                      default='',   help='')
    parser.add_option('--valid_fname',          dest='valid_fname',         metavar='FILE',
                      default='',   help='')
    parser.add_option('--test_fname',           dest='test_fname',          metavar='FILE',
                      default='',   help='')
    parser.add_option('--word_vector_cache',    dest='word_vector_cache',   metavar='FILE',
                      default='',   help='dir for caching pretrained word vectors')
    parser.add_option('--result_dir',           dest='result_dir',          metavar='FILE',     
                      default='',   help='dir to store results')
    parser.add_option('--log_name',             dest='log_name', metavar='FILE',
                      default='pre_lm0.log')
    parser.add_option('--save_model',           action='store_true',        dest='save_model',
                      default=False)
    
    # optmizer & initializer
    parser.add_option('--initializer',          dest='initializer',
                      default='normal',     help='[glorot,constant,uniform,normal]')
    parser.add_option('--optimizer',            dest='optimizer',
                      default='adagrad',    help='[sgd,momentum,adam,adadelta,adagrad]')
    # model params
    parser.add_option("--lm_pretrain",          action='store_true', dest="lm_pretrain",    default=False)
    parser.add_option("--tie_weights",          action='store_true', dest="tie_weights",    default=False)
    parser.add_option('--lm_word_dim',          type='int',     dest='lm_word_dim',         default=100)
    parser.add_option('--lm_pos_dim',           type='int',     dest='lm_pos_dim',          default=0)  # 50
    parser.add_option('--lm_lstm_dim',          type='int',     dest='lm_lstm_dim',         default=100)
    parser.add_option('--lm_nlayers',           type='int',     dest='lm_nlayers',          default=2)
    parser.add_option('--batchsize',            type='int',     dest='batchsize',           default=64)
    # training options
    parser.add_option('--decay_every',      type='int', dest='decay_every',     default=5000)
    parser.add_option('--epochs',               type='int', dest='epochs',          default=50)
    parser.add_option('--print_every',          type='int', dest='print_every',     default=20)
    # optimization misc
    parser.add_option('--lr',               type='float', dest='lr',            default=0.01) # 1e-3
    # parser.add_option('--lrdecay',      type='float', dest='decay',         default=0.75)
    parser.add_option('--lm_dropout',   type='float', dest='lm_dropout',     default=0.2)
    parser.add_option('--l2_reg',       type='float', dest='l2_reg',        default=1e-4)
    parser.add_option('--clip',         type='float', dest='clip',          default=0.25)
    # limit length options
    parser.add_option('--train_max_length', type='int', dest='train_max_length', default=20)
    parser.add_option('--min_freq',         type='int', dest='min_freq',        default=2)
    parser.add_option('--gpu_id',       type='int',     dest='gpu_id',      default=-1)

    (options, args) = parser.parse_args()
    s = Session(options)
    s.train_lm()