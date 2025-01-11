#include "main.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
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


std::vector<std::pair<unsigned int, std::string>> lexicon;


std::vector<float> encode_string(const std::string& input_string) {
    std::vector<float> encoded_floats;
    
    std::vector<std::string> words = Explode(input_string, ' ');
    
    unsigned int tokexIndex = 1;
    
    // Parse words
    for (unsigned int i=0; i < words.size(); i++) {
        
        bool isFound = false;
        
        std::string word = words[i];
        
        unsigned int numberOfLexes = lexicon.size();
        for (unsigned int a=0; a < numberOfLexes; a++) {
            
            std::pair<unsigned int, std::string> lexword = lexicon[a];
            
            // Word exists in the lexicon
            if (lexword.second != word) 
                continue;
            
            // Add existing word
            unsigned int token = lexword.first;
            
            float normalized_value = static_cast<float>(token) / 128.0f;
            encoded_floats.push_back(normalized_value);
            
            isFound = true;
            break;
        }
        
        if (isFound == true) 
            continue;
        
        // Add new word
        
        std::pair<unsigned int, std::string> token(tokexIndex, word);
        
        float normalized_value = static_cast<float>(tokexIndex) / 128.0f;
        
        encoded_floats.push_back(normalized_value);
        
        tokexIndex++;
        
    }
    
    return encoded_floats;
}



std::string decode_string(const std::vector<float>& encoded_floats) {
    std::string decoded_string;
    
    for (unsigned int i=0; i < encoded_floats.size(); i++) {
        
        unsigned int decodedValue = static_cast<unsigned int>(encoded_floats[i] * 128.0f);
        
        if ((decodedValue-1) > lexicon.size()) 
            continue;
        
        std::pair<unsigned int, std::string> lexpair = lexicon[decodedValue - 1];
        
        decoded_string += lexpair.second + " ";
        
    }
    
    return decoded_string;
}







int main() {
    
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Create a neural network
    NeuralNetwork nnet;
    
    // Add layers to the neural network
    nnet.AddNeuralLayer(32, 1);
    
    nnet.AddNeuralLayer(32, 32);
    nnet.AddNeuralLayer(32, 32);
    nnet.AddNeuralLayer(32, 32);
    
    nnet.AddNeuralLayer(1, 32);
    
    
    // Train the model
    
    std::cout << "Parsing..." << std::endl;
    
    std::string sourceText = "The black cat jumped over the lazy dog";
    std::vector<std::string> sourceSplit = Explode(sourceText, ' ');
    
    std::vector<TrainingSet> trainingBook;
    
    for (unsigned int i=0; i < sourceSplit.size() - 1; i++) {
        
        std::string wordA = sourceSplit[i];
        std::string wordB = sourceSplit[i + 1];
        
        TrainingSet tsa;
        tsa.input  = encode_string(wordA);
        tsa.target = encode_string(wordB);
        trainingBook.push_back(tsa);
        
    }
    
    
    
    std::cout << "Training..." << std::endl;
    
    for (int epoch = 0; epoch < 100; epoch++) {
        
        for (unsigned int i=0; i < trainingBook.size(); i++) 
            nnet.Train(trainingBook[i], 0.018f);
        
    }
    
    // Test dataset
    std::string outputText = "The";
    std::string prevText   = "The";
    
    std::cout << "Output" << std::endl;
    
    while(1) {
        
        std::vector<float> dataset = encode_string(outputText);
        
        nnet.FeedForward(dataset);
        std::vector<float> results =  nnet.GetResults();
        
        outputText = decode_string(results);
        
        std::string finalOutput = outputText;
        
        // Remove trailing spaces
        finalOutput.erase( outputText.find_last_not_of(" \t\n\r\f\v") + 1 );
        
        // Check if the text has gone out of range
        for (unsigned int a=0; a < outputText.size(); a++) {
            
            if ((outputText[a] >= 0x20) && (outputText[a] <= 0x7a)) {
                
                std::cout << outputText[a];
                
                if (prevText == outputText) 
                    continue;
                
                
                
                TrainingSet trainingSet;
                trainingSet.input  = encode_string(outputText);
                trainingSet.target = encode_string(prevText);
                
                nnet.Train(trainingSet, 0.0001f);
                
                prevText = outputText;
                prevText.resize(8, ' ');
                
                
                
                
            } else {
                
            }
            
        }
        
        std::cout << " ";
        
    }
    
    std::cout << std::endl << std::endl << std::endl;
    
    return 0;
}

