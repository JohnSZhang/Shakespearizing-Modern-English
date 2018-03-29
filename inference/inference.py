import tensorflow as tf
import numpy as np
import pickle
import sys
import configuration as config
import mt_model_pointer as model
from prepro import PreProcessing
from keras.preprocessing.sequence import pad_sequences

# To Perform Inference on file: python inference.py input.txt output.txt
# currently hardcoded to use pointer model 6 with max batchsize 32
# Make sure you have pointer_model6 files in the inference/data/models directory

def main():
    input_file_loc = sys.argv[1]
    print 'inferencing from file: ', input_file_loc
    input_file = open(input_file_loc, 'r')
    input_texts = input_file.readlines()
    print 'input text:', input_texts
    
    # Set up seeds
    tf.set_random_seed(10)
    np.random.seed(1)

    # params from config
    params = {}
    params['embeddings_dim'] = config.embeddings_dim
    params['lstm_cell_size'] = config.lstm_cell_size
    params['max_input_seq_length'] = config.max_input_seq_length
    params[
        'max_output_seq_length'] = config.max_output_seq_length - 1  # inputs are all but last element, outputs are al but first element
    params['batch_size'] = config.batch_size
    params['pretrained_embeddings'] = config.use_pretrained_embeddings
    params['share_encoder_decoder_embeddings'] = config.share_encoder_decoder_embeddings
    params['use_pointer'] = config.use_pointer
    params['pretrained_embeddings_path'] = config.pretrained_embeddings_path
    params['pretrained_embeddings_are_trainable'] = config.pretrained_embeddings_are_trainable
    params['use_additional_info_from_pretrained_embeddings'] = config.use_additional_info_from_pretrained_embeddings
    params['max_vocab_size'] = config.max_vocab_size
    params['do_vocab_pruning'] = config.do_vocab_pruning
    params['use_reverse_encoder'] = config.use_reverse_encoder
    params['use_sentinel_loss'] = config.use_sentinel_loss
    params['lambd'] = config.lambd
    params['use_context_for_out'] = config.use_context_for_out

    # Get our preprocessing object for word dicts, etc
    data_src = "data/"
    preprocessing = pickle.load(open(data_src + "preprocessing.obj","r") )
    params['vocab_size'] = preprocessing.vocab_size
    params['preprocessing'] = preprocessing


    # Pretrained embeddings
    pretrained_embeddings = pickle.load(open(params['pretrained_embeddings_path'],"r"))
    word_to_idx = preprocessing.word_to_idx
    encoder_embedding_matrix = np.random.rand(params['vocab_size'], params['embeddings_dim'] )
    decoder_embedding_matrix = np.random.rand(params['vocab_size'], params['embeddings_dim'] )
    not_found_count = 0
    for token,idx in word_to_idx.items():
        if token in pretrained_embeddings:
            encoder_embedding_matrix[idx]=pretrained_embeddings[token]
            decoder_embedding_matrix[idx]=pretrained_embeddings[token]
    params['encoder_embeddings_matrix'] = encoder_embedding_matrix
    params['decoder_embeddings_matrix'] = decoder_embedding_matrix

    if params['use_additional_info_from_pretrained_embeddings']:
        additional_count=0
        tmp=[]
        for token in pretrained_embeddings:
            if token not in preprocessing.word_to_idx:
                preprocessing.word_to_idx[token] = preprocessing.word_to_idx_ctr
                preprocessing.idx_to_word[preprocessing.word_to_idx_ctr] = token
                preprocessing.word_to_idx_ctr+=1
                tmp.append(pretrained_embeddings[token])
                additional_count+=1
        params['vocab_size'] = preprocessing.word_to_idx_ctr
        tmp = np.array(tmp)
        encoder_embedding_matrix = np.vstack([encoder_embedding_matrix,tmp])
        decoder_embedding_matrix = np.vstack([decoder_embedding_matrix,tmp])



    pp = pickle.load(open(data_src + "preprocessing.obj","r") )

    # the load data step
    input_texts = pp.preprocess(input_texts)
    sequences_input = []
    for i in input_texts:
        tmp = [word_to_idx[pp.sent_start]]
        for token in i:
            if token not in word_to_idx:
                tmp.append(word_to_idx[pp.unknown_word])
            else:
                tmp.append(word_to_idx[token])
        tmp.append(word_to_idx[pp.sent_end])
        sequences_input.append(tmp)

    sequences_input = pad_sequences(sequences_input, maxlen=config.max_input_seq_length,
                                   padding="pre", truncating="post")

    sequences_input = np.array(sequences_input)

    # Perform Greedy Inference
    inference_type = 'greedy'
    pointer_model = model.RNNModel(buckets_dict=None, mode='inference', params=params)
    # pointer_model.token_emb_mat =
    optimizer_type = "adam"
    lr = 0.001
    encoder_outputs = pointer_model.getEncoderModel(params,
                                    mode='inference', reuse=False)
    pointer_model.getDecoderModel(params,
            encoder_outputs, is_training=False, mode='inference', reuse=False )

    # Bring saved model back
    sess = tf.Session()
    saver = tf.train.Saver()
    # TODO: define own saved model path
    saved_model_path = "data/models/pointer_model6.ckpt"
    saver.restore(sess, saved_model_path)


    encoder_outputs = pointer_model.getEncoderModel(params, mode='inference', reuse= True)
    decoder_outputs_inference, encoder_outputs, alpha_inference = \
        pointer_model.getDecoderModel(params, encoder_outputs, is_training=False,
                                      mode='inference', reuse= True )



    # TODO: change batch size to custom ones from input
    batch_size = 32
    input_length = sequences_input.shape[0]
    gap = batch_size - input_length
    for j in range(gap):
        sequences_input = np.vstack((sequences_input, sequences_input[0]))

    feed_dict = {
        pointer_model.token_lookup_sequences_placeholder_inference: sequences_input
    }

    decoder_output, encoder_output, alpha_inf = np.array(
        sess.run([decoder_outputs_inference,
                  encoder_outputs, alpha_inference], feed_dict=feed_dict))

    # vectorized encoder sentence
    # print 'encoder outputs', encoder_output

    # decoder output that we can reverse into sentences
    print 'decoder outputs', decoder_output

    reverse_dict = pp.idx_to_word
    output = ""
    for o in decoder_output[:input_length]:
        print [reverse_dict[j] for j in o]

    reverse_dict = pp.idx_to_word
    output = ""
    for o in sequences_input[:input_length]:
        print [reverse_dict[j] for j in o]
        
    output_file_loc = sys.argv[2]
    output_file = open(output_file_loc, "w")
    print "writing to output file: ", output_file_loc
    reverse_dict = pp.idx_to_word
    for o in decoder_output[:input_length]:
        line = ' '.join([reverse_dict[j] for j in o]) + "\n" 
        output_file.write(line)
    output_file.close()
        
if __name__ == "__main__":
    main()
