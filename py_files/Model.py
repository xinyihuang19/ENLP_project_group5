import json
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizer, BertModel
import tensorflow as tf
import keras
import numpy as np

#####
# Global variables
#####

# Check if CUDA can be used to speed up training/reasoning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CE
# Load BERT-large tokenizer and BERT-Large model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)  # Make sure the model is on the correct device

# Define BiGRU layer for CE
hidden_size = 1024  # For BERT-Large，the hidden_size should be 1024
bigru_layer = nn.GRU(input_size=1024, hidden_size=hidden_size, bidirectional=True, batch_first=True).to(device)

# NE
# Character-to-index mapping
char_to_index = {str(i): i for i in range(10)}
char_to_index['.'] = 10  # The index for the decimal point

# Maximum numeric length and character dimension
max_num_length = 10
char_dim = 11

# Initialize BiGRU for NE
input_size_NE = char_dim
hidden_size = 1024
bigru_model = nn.GRU(input_size=input_size_NE, hidden_size=hidden_size, bidirectional=True, batch_first=True)

# Define a function to encode text using BERT and BiGRU
def encode_with_ce(texts):
    # Encode the texts
    encoded_input = tokenizer(texts, return_tensors='pt',padding='max_length', truncation=True, max_length=512,
                              add_special_tokens=True)

    # Make sure the input is also on the correct device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Get the embedding using BERT-Large
    with torch.no_grad():
        output = bert_model(**encoded_input)

    # The BERT model outputs a tuple, and we are interested in the first element - the hidden state
    embeddings = output.last_hidden_state

    # Pass the embed to BiGRU
    bigru_output, _ = bigru_layer(embeddings)

    return bigru_output

# Convert answer's number into representations
def encode_and_process_number(number, max_num_length=10, char_dim=11, bigru_model=bigru_model):
    # Create an all-zero tensor
    encoded = torch.zeros(max_num_length, char_dim)

    # Calculate the left fill amount
    padding_size = max_num_length - len(number)

    # Fill encoding according to character
    for i, char in enumerate(number):
        if char in char_to_index:
            encoded[padding_size + i, char_to_index[char]] = 1

    # Add batch dimension
    encoded = encoded.unsqueeze(0)  # Make the tensor shape [1, max_num_length, char_dim]

    # Input the encoded tensor into BiGRU
    bigru_output, _ = bigru_model(encoded)

    # Return the output of BiGRU
    return bigru_output

# Converts the correct answer index to a unique thermal encoding
def one_hot_encode(index, num_classes):
    encoding = [0] * num_classes
    encoding[index] = 1
    return encoding

def main():
    ##########
    # 1. Load the data
    ##########

    # Import the training data
    with open('/kaggle/input/nquad-dataset/NQuAD_train_first_10k.json', 'r', encoding='utf-8') as file:
        data_train = json.load(file)

    # Import the testing data
    with open('/kaggle/input/nquad-dataset/NQuAD_test_first_2k.json', 'r', encoding='utf-8') as file:
        data_test = json.load(file)

    ##########
    # 2. Generate question representations for training data
    ##########

    # Prepare training data and one-hot labels lists
    question_representations_train = []
    one_hot_labels_train = []

    # Iterate each sample in training data
    for sample in data_train[:2]:
        # CE output list
        ce_output_list = []
        # NE output list
        ne_output_list = []

        # 2.1 CE
        # Process question stem
        stem_result = encode_with_ce(sample['question_stem'])
        ce_output_list.append(stem_result)

        # Process sentences_containing_the_numeral_in_answer_options
        for sentence_list in sample["sentences_containing_the_numeral_in_answer_options"]:

            if len(sentence_list) > 1:
                combined_sentence = '。'.join(sentence.strip() for sentence in sentence_list)
                # print(combined_sentence)
                result = encode_with_ce(combined_sentence)
                ce_output_list.append(result)
            else:
                # print(sentence_list[0])
                result = encode_with_ce(sentence_list[0].strip())
                ce_output_list.append(result)

        # for each in ce_output_list:
            # print(each.shape)

        # 2.2 NE
        numbers = sample["answer_options"]
        for number in numbers:
            result = encode_and_process_number(number)
            ne_output_list.append(result)
        #             print(result.shape)

        # for each in ne_output_list:
        #     print(each.shape)

        # 2.3 Concatenate
        all_output_list = ce_output_list + ne_output_list
        all_numpy_arrays_list = [tensor.detach().cpu().numpy() for tensor in all_output_list]

        # Use tf. Keras. The layers. Concatenate to joining together all these tensor
        concat_layer = tf.keras.layers.Concatenate(axis=1)
        concatenated_tensors = concat_layer(all_numpy_arrays_list)
        # print("concatenated_tensors.shape")
        # print(concatenated_tensors.shape)

        # 2.4 Apply global average pooling
        global_average_layer = tf.keras.layers.GlobalAveragePooling1D()
        pooled_tensor = global_average_layer(concatenated_tensors)
        # print("pooled_tensor.shape")
        # print(pooled_tensor.shape)

        # Convert it to a TensorFlow tensor
        pooled_tensor = tf.convert_to_tensor(pooled_tensor)

        # Add pooled tensor to question_representations
        question_representations_train.append(pooled_tensor)

        # 2.5 Convert index of answer into one-hot vector
        correct_answer_index = [sample['ans']]
        # Converts the correct answer index to TensorFlow's uniquely thermal coded tensor
        one_hot_label = tf.one_hot(correct_answer_index, depth=4)
        #         print(one_hot_label)
        #         print(type(one_hot_label))
        one_hot_labels_train.append(one_hot_label)
    #     print()
    #
    # print(len(question_representations_train))
    # print(len(one_hot_labels_train))


    ##########
    # 3. Make MLP model and put question representations and one-hot label list into MLP model
    ##########

    # Construct a MLP model
    mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(2048,)),
        tf.keras.layers.Dropout(0.3),  # Add dropout
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Compile model
    mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    mlp.fit(question_representations_train, one_hot_labels_train, epochs=10, batch_size=32)

    #####
    # 4. Prepare testing data and make predictions
    #####

    # Prepare training data and one-hot labels lists
    question_representations_test = []
    one_hot_labels_test = []

    # Iterate each sample in training data
    for sample in data_test[:1]:
        # CE output list
        ce_output_list = []
        # NE output list
        ne_output_list = []

        # 2.1 CE
        # Process question stem
        stem_result = encode_with_ce(sample['question_stem'])
        ce_output_list.append(stem_result)

        # Process sentences_containing_the_numeral_in_answer_options
        for sentence_list in sample["sentences_containing_the_numeral_in_answer_options"]:
            if len(sentence_list) > 1:
                combined_sentence = '。'.join(sentence.strip() for sentence in sentence_list)
                print(combined_sentence)
                result = encode_with_ce(combined_sentence)
                ce_output_list.append(result)
            else:
                print(sentence_list[0])
                result = encode_with_ce(sentence_list[0].strip())
                ce_output_list.append(result)
        for each in ce_output_list:
            print(each.shape)

        # 2.2 NE
        numbers = sample["answer_options"]
        for number in numbers:
            result = encode_and_process_number(number)
            ne_output_list.append(result)
        #             print(result.shape)

        for each in ne_output_list:
            print(each.shape)

        # 2.3 Concatenate
        all_output_list = ce_output_list + ne_output_list
        all_numpy_arrays_list = [tensor.detach().cpu().numpy() for tensor in all_output_list]

        # Use tf. Keras. The layers. Concatenate to joining together all these tensor
        concat_layer = tf.keras.layers.Concatenate(axis=1)
        concatenated_tensors = concat_layer(all_numpy_arrays_list)
        print("concatenated_tensors.shape")
        print(concatenated_tensors.shape)

        # 2.4 Apply global average pooling
        global_average_layer = tf.keras.layers.GlobalAveragePooling1D()
        pooled_tensor = global_average_layer(concatenated_tensors)
        print("pooled_tensor.shape")
        print(pooled_tensor.shape)

        # Convert it to a TensorFlow tensor
        pooled_tensor = tf.convert_to_tensor(pooled_tensor)

        # Add pooled tensor to question_representations
        question_representations_test.append(pooled_tensor)

        # 2.5 Convert index of answer into one-hot vector
        correct_answer_index = [sample['ans']]
        # Converts the correct answer index to TensorFlow's uniquely thermal coded tensor
        one_hot_label = tf.one_hot(correct_answer_index, depth=4)
        #         print(one_hot_label)
        #         print(type(one_hot_label))
        one_hot_labels_test.append(one_hot_label)
        print()

    print(len(question_representations_test))
    print(len(one_hot_labels_test))

    # Evaluate the model
    # Evaluate the model using the test set data question_representations_test and one_hot_labels_test
    loss, accuracy = mlp.evaluate(question_representations_test, one_hot_labels_test)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()