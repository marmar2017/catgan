# for course review (23k+), model trained at epoch 90 is selected as we need to focus on neutral and negative review 
# based on BLEU and nll_div of neutral and negative reivews, we select epoch 90
# model locates at 
# /home/user1/Ru_experiement/TextGAN-PyTorch/save/20210331/cr150/catgan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl37_temp100_lfd0.001_T0331_1910_23/models/gen_ADV_00090.pt


# there are 18476 positive reviews and 2316 negative reviews and 1145 neutral reviews
# so we need to generate 1254, 17684, and 1145 reivews for respective category
import torch
from models.CatGAN_G import CatGAN_G
import config as cfg
from utils.text_process import load_dict
from utils.text_process import write_tokens
from utils.text_process import tensor_to_tokens
from datetime import datetime

def generate_text(model, model_path,balanced_length, current_lengths, start_letter, dict_path):
    _, idx2word_dict = load_dict(dict_path)
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model)
    model.eval()
    model.cuda()
    for label in current_lengths.keys(): 
        current_time = datetime.now().strftime("%H:%M:%S")
        print('%s  generating %d %d reviews...'%(current_time, balanced_length-current_lengths[label], label))   
        samples = model.sample(
                    num_samples=balanced_length-current_lengths[label], 
                    batch_size=64, 
                    one_hot=False, 
                    label_i=label,
                    start_letter=start_letter)
        save_sample_path = './generated_text/' + 'samples_{}_{}.txt'.format(dict_path, label)
        write_tokens(save_sample_path, tensor_to_tokens(samples, idx2word_dict))
def generate_cr_text():
    k_label = 3
    mem_slots = 1
    num_heads = 2
    head_size= 512
    gen_embed_dim = 32
    gen_hidden_dim = 32
    vocab_size= 6950
    max_seq_len= 37
    padding_idx = 0
    cuda = int(True)
    start_letter = 1

    model = CatGAN_G(k_label, mem_slots, num_heads, head_size, gen_embed_dim,
                                gen_hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=cuda)
    trained_model_path = '/home/user1/Ru_experiement/TextGAN-PyTorch/save/20210331/cr150/catgan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl37_temp100_lfd0.001_T0331_1910_23/models/gen_ADV_00090.pt'
    current_categoty_lengths = {0:18746, 1:1145,2:2316}

    generate_text(
    model=model, 
    model_path=trained_model_path,
    balanced_length=20000,
    current_lengths=current_categoty_lengths,
    start_letter = start_letter,
    dict_path = 'cr150'
    )

def generate_sr250_text():
    k_label = 3
    mem_slots = 1
    num_heads = 2
    head_size= 512
    gen_embed_dim = 32
    gen_hidden_dim = 32
    vocab_size=26338
    max_seq_len= 63
    padding_idx = 0
    cuda = int(True)
    start_letter = 1

    model = CatGAN_G(k_label, mem_slots, num_heads, head_size, gen_embed_dim,
                                gen_hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=cuda)
    trained_model_path = '/home/user1/Ru_experiement/TextGAN-PyTorch/save/20210402/sr250/catgan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl63_temp100_lfd0.001_T0402_1237_21/models/gen_ADV_00050.pt'
    current_categoty_lengths = {0:74191, 1:3181,2:2602}

    generate_text(
        model=model, 
        model_path=trained_model_path,
        balanced_length=80000,
        current_lengths=current_categoty_lengths,
        start_letter = start_letter,
        dict_path = 'sr250'
    )
generate_sr250_text()

# /home/user1/Ru_experiement/TextGAN-PyTorch/save/20210328/sr/sentigan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl70_temp1_lfd0.0_T0328_1955_49/samples/samples_d0_ADV_00000.txt