#include "main.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>


std::vector<std::string> Explode(const std::string& value, const char character) {
	std::vector<std::string> result;
    std::istringstream iss(value);
    
    for (std::string token; std::getline(iss, token, character); ) {
        
        if (std::move(token) == "") 
            continue;
        
        result.push_back(std::move(token));
    }
    return result;
}



std::vector<float> encode_string(const std::string& input_string) {
    std::vector<float> encoded_floats;
    
    for (unsigned int i=0; i < input_string.size(); i++) {
        uint8_t char_code = input_string[i];
        float normalized_value = static_cast<float>(char_code) / static_cast<float>(0xff);
        encoded_floats.push_back(normalized_value);
    }
    
    return encoded_floats;
}



std::string decode_string(const std::vector<float>& encoded_floats) {
    std::string decoded_string;
    
    for (size_t i = 0; i < encoded_floats.size(); ++i) {
        
        uint8_t char_code = std::round(encoded_floats[i] * 0xff);
        
        decoded_string.push_back( char_code );
    }
    
    return decoded_string;
}







int main() {
    
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Create a neural network
    NeuralNetwork nnet;
    
    // Add layers to the neural network
    nnet.AddNeuralLayer(64, 8);
    
    nnet.AddNeuralLayer(64, 64);
    nnet.AddNeuralLayer(32, 64);
    nnet.AddNeuralLayer(32, 32);
    nnet.AddNeuralLayer(32, 32);
    
    nnet.AddNeuralLayer(8, 32);
    
    
    // Train the model
    
    TrainingSet tsa;
    tsa.input  = encode_string("The     ");
    tsa.target = encode_string("black   ");
    
    TrainingSet tsb;
    tsb.input  = encode_string("black   ");
    tsb.target = encode_string("cat     ");
    
    TrainingSet tsc;
    tsc.input  = encode_string("cat     ");
    tsc.target = encode_string("jumped  ");
    
    TrainingSet tsd;
    tsd.input  = encode_string("jumped  ");
    tsd.target = encode_string("over    ");
    
    TrainingSet tse;
    tse.input  = encode_string("over    ");
    tse.target = encode_string("the     ");
    
    TrainingSet tsf;
    tsf.input  = encode_string("the     ");
    tsf.target = encode_string("lazy    ");
    
    TrainingSet tsg;
    tsg.input  = encode_string("lazy    ");
    tsg.target = encode_string("dog     ");
    
    TrainingSet tsh;
    tsh.input  = encode_string("dog     ");
    tsh.target = encode_string("The     ");
    
    std::vector<TrainingSet> trainingBook;
    trainingBook.push_back(tsa);
    trainingBook.push_back(tsb);
    trainingBook.push_back(tsc);
    trainingBook.push_back(tsd);
    trainingBook.push_back(tse);
    trainingBook.push_back(tsf);
    trainingBook.push_back(tsg);
    trainingBook.push_back(tsh);
    
    
    
    for (int epoch = 0; epoch < 128000; epoch++) {
        
        for (unsigned int i=0; i < trainingBook.size(); i++) 
            nnet.Train(trainingBook[i], 0.018f);
        
    }
    
    
    // Test dataset
    std::string reactivationText = "The     ";
    
    std::string outputText = "The     ";
    
    for (unsigned int i=0; i < 10; i++) {
        
        while(1) {
            
            std::vector<float> dataset = encode_string(outputText);
            
            nnet.FeedForward(dataset);
            std::vector<float> results =  nnet.GetResults();
            
            outputText = decode_string(results);
            
            std::string finalOutput = outputText;
            
            // Remove trailing spaces
            finalOutput.erase( outputText.find_last_not_of(" \t\n\r\f\v") + 1 );
            
            std::cout << finalOutput << " ";
            
            // Check if the text has gone out of range
            for (unsigned int a=0; a < outputText.size(); a++) {
                if (outputText[a] < 0x20) {
                    
                    outputText = reactivationText;
                    
                    break;
                }
                
            }
            
        }
        
    }
    
    std::cout << std::endl << std::endl << std::endl;
    
    return 0;
}


