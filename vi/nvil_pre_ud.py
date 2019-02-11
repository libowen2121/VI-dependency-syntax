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
    parser.add_option('--cluster',              dest='cluster',             action='store_true',
                      default=False)
    parser.add_option('--cluster_fname',        dest='cluster_fname',        metavar='FILE',
                      default='',   help='dir pretrained word cluster files')
    parser.add_option('--word_vector_cache',    dest='word_vector_cache',   metavar='FILE',
                      default='',   help='dir for caching pretrained word vectors')
    parser.add_option('--result_dir',           dest='result_dir',          metavar='FILE',     
                      default='',   help='dir to store results')
    parser.add_option('--encoder_fname',        dest='encoder_fname',       metavar='FILE',     
                      default='',   help='dir to store encoder models') # for loading
    parser.add_option('--decoder_fname',        dest='decoder_fname',       metavar='FILE',     
                      default='',   help='dir to store decoder models') # for loading
    parser.add_option('--lm_fname',             dest='lm_fname',            metavar='FILE',     
                      default='',   help='dir to store baseline models')    # for loading
    parser.add_option('--log_name', dest='log_name', metavar='FILE',
                      default='nvil_test.log')  
    parser.add_option('--save_model', action='store_true', dest='save_model', default=False)
    
    # optmizer & initializer
    parser.add_option('--initializer',          dest='initializer',
                      default='glorot',     help='[glorot,constant,uniform,normal]')
    parser.add_option('--optimizer',            dest='optimizer',
                      default='adagrad',    help='[sgd,momentum,adam,adadelta,adagrad]')
    # model params
    parser.add_option("--pretrain_word_dim",type="int",     dest="pretrain_word_dim",    default=300)    
    parser.add_option('--word_dim',         type='int',     dest='word_dim',        default=80)
    parser.add_option('--pos_dim',          type='int',     dest='pos_dim',         default=80)
    parser.add_option('--action_dim',       type='int',     dest='action_dim',      default=0)  # 32
    parser.add_option('--enc_lstm_dim',     type='int',     dest='enc_lstm_dim',    default=64)
    parser.add_option('--dec_lstm_dim',     type='int',     dest='dec_lstm_dim',    default=64)
    parser.add_option('--nlayers',          type='int',     dest='nlayers',         default=1)
    # training options
    parser.add_option('--epochs',           type='int', dest='epochs',          default=10)
    parser.add_option('--print_every',      type='int', dest='print_every',     default=10)
    parser.add_option('--save_every',       type='int', dest='save_every',      default=10)
    # optimization misc
    parser.add_option('--lr',           type='float',   dest='lr',            default=0.01) # 0.01
    parser.add_option('--clip',         type='float',   dest='clip',          default=0.5)
    
    parser.add_option('--enc_dropout',  type='float', dest='enc_dropout',   default=0.5)
    parser.add_option('--dec_dropout',  type='float', dest='dec_dropout',   default=0.5)
    parser.add_option('--l2_reg',       type='float', dest='l2_reg',        default=1e-4)
    # limit length options
    parser.add_option('--train_max_length', type='int', dest='train_max_length', default=10)
    parser.add_option('--gpu_id',       type='int',     dest='gpu_id',      default=-1)
    # test options
    parser.add_option('--output_tree', action='store_true', dest='output_tree', default=False)
    
    # for pr
    parser.add_option('--pr_initializer',       dest='pr_initializer',
                      default='normal',     help='[glorot,constant,uniform,normal]')
    parser.add_option('--pr_optimizer',         dest='pr_optimizer',
                      default='sgd',        help='[sgd,momentum,adam,adadelta,adagrad]')
    parser.add_option('--epsilon',              type='float',   dest='epsilon',          default=0.1)
    parser.add_option('--pr_fname',             dest='pr_fname',           metavar='FILE',
                      default='',   help='')
    parser.add_option('--mc_samples',           type='int',     dest='mc_samples',      default=20)
    parser.add_option('--batchsize',            type='int',     dest='batchsize',       default=1)
    parser.add_option('--nvil_batchsize',       type='int',     dest='nvil_batchsize',  default=1)    
    parser.add_option('--min_freq',             type='int',     dest='min_freq',        default=2)
    parser.add_option('--seed',             type='int',     dest='seed',        default=1000)
    
    parser.add_option('--language',           dest='language',          metavar='FILE',
                      default='', help='')
    (options, args) = parser.parse_args()    
    s = Session(options)
    s.nvil_pr_pretrain_ud()