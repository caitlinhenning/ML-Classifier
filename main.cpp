/* A program to comb through posts on Piazza (a platform for students to ask 
project/homework questions to staff and other peers) from previous terms (which 
are already tagged according to topic) and learn which words go with which topics. 
This approach is called (supervised) machine learning. Once we’ve trained the 
classifier on some set of Piazza posts, we can apply it to new ones written in 
the future. */

#include <map>
#include <set>
#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <math.h>
#include "csvstream.h"

using namespace std;

class Classifier {
    private:
    double total_posts; // Total number of posts in the entire training set
    double vocab_size;  // Number of unique words in the entire training set
    map<string,double> word_count; // For each word w, the num posts in set containing w
    map<string,double> label_count; // For each label C, num posts in set labeled C
    map<string, map<string,double>> C_w_count; // Num posts with label C that contain w
    map<string, string> post; // <column name, cell datum>
    map<string,string> correct_post; // for checking correctness: <post, correct label>
                                

    public:

    // RETURNS: a set containing the unique "words" in the original
    //          string, delimited by whitespace
    // EFFECTS: -
    // MODIFIES: -
    set<string> unique_words(const string &str) {
        istringstream source(str);
        set<string> words;
        string word;

        // Read word by word from the stringstream and insert into the set
        while (source >> word) {
            words.insert(word);
        }
        return words;
    }

    int get_total_posts() {
        return int (total_posts);
    }

    double get_vocab_size() {
        return vocab_size;
    }

    // RETURNS: double representing log prior 
        // = log(num training posts with label C / num training posts)
    // EFFECTS: -
    // MODIFIES: -
    double calc_log_prior(string label) {
        return log(label_count[label]/total_posts);
    }

    // RETURNS: double representing log likelihood 
    // EFFECTS: -
    // MODIFIES: -
    double calc_log_likelihood(string label, string word) {
        // If w does not occur in posts labeled C but occurs in training set
        if(C_w_count[label][word] == 0 && word_count[word] != 0) {
            return log(word_count[word] / total_posts);
        }
        // If w does not occur anywhere in training set
        else if (word_count[word] == 0) {
            return log(1 / total_posts);
        }
        // log(num posts with label C containing w / num training posts)
        else {
            return log(C_w_count[label][word]/label_count[label]);
        }
    }

    //calculate log probability score
    //sum of log-prior and log likelihoods of each unique word in post
    // RETURNS: double representing log probability score
        // calculated from summing log likelihood of each word in the post
    // EFFECTS: calls calc_log_prior(string label)
    // MODIFIES: -
    double calc_log_prob_score(string label) {
        string content = correct_post["content"];
        set<string> unique_post_words = unique_words(content);
        double log_prior = calc_log_prior(label);
        double log_prob_score = log_prior;
        
        for(auto word : unique_post_words) {
            log_prob_score += calc_log_likelihood(label, word);
        }
        return log_prob_score;
    }

    // RETURNS: -
    // EFFECTS: prints training data
    // MODIFIES: -
    void print_label_content(int argc, char* argv[]) {
        const string debug = argv[3];
        string train_file = argv[1];
        csvstream csvin(train_file);
        cout << "training data:" << endl;
        // For each post, print "label = ___, content = ____"
        while(csvin >> post) {  
            cout << "  label = " << post["tag"] << ", content = " 
            << post["content"] << endl;
        }
    }

    // RETURNS: -
    // EFFECTS: prints the classes in the training data and num examples for each;
        // prints for each label, and for each word that occurs for that label: 
        // the number of posts with that label that contained the word, 
        // and the log-likelihood of the word given the label.
    // MODIFIES: 
    void print_debug_data(int argc, char* argv[]) {
        const string debug = argv[3];
        
        cout << "classes:" << endl;
        for(auto label : label_count) {
            cout << "  " << label.first << ", " << label.second << " examples, " 
                << "log-prior = " << calc_log_prior(label.first) << endl;
        }
        
        cout << "classifier parameters:" << endl;
        for(const auto& label : C_w_count) {
            for(const auto& word_pair : label.second) {
                cout << "  " << label.first << ":" << word_pair.first << 
                    ", count = " << word_pair.second << ", log-likelihood = "
                    << calc_log_likelihood(label.first, word_pair.first) << endl;
            }
        }
        // extra blankline
        cout << "\n";
        
    }

    // RETURNS: -
    // EFFECTS: calls unique_words(const string &str)
    // MODIFIES: label_count, word_count, C_w_count, total_posts, vocab_size
    void train_classifier(int argc, char *argv[]) {
        string train_file = argv[1];
        csvstream csvin(train_file);

        set<string> unique_words_set;

        total_posts = 0;
        while(csvin >> post) {
            label_count[post["tag"]] += 1;
            unique_words_set = unique_words(post["content"]);
            for(auto word : unique_words_set) {
                word_count[word] += 1;
                C_w_count[post["tag"]][word] += 1;
            }
            
            total_posts++;
        }
        
        vocab_size = word_count.size();
    }

    // RETURNS: pair<string,double> representing post prediction and max probability score
    // EFFECTS: calls calc_log_prob_score(string label)
    // MODIFIES: -
    pair<string,double> predict_label() {
        vector<pair<string,double>> prob_scores;
        // For each label in the training data, find the log prob score of the post
        for(const auto& label : C_w_count) {
            prob_scores.push_back({label.first,calc_log_prob_score(label.first)});
        }
        double max_score = prob_scores[0].second;
        string prediction = prob_scores[0].first;
        for(int i = 0; i < prob_scores.size(); i++) {
            if(prob_scores[i].second > max_score) {
                max_score = prob_scores[i].second;
                prediction = prob_scores[i].first;
            }
        }
        return {prediction, max_score};
    }

    // RETURNS: a pair of ints <number of correctly labeled posts, number of posts>
    // EFFECTS: prints line-by-line, the “correct” label, the predicted label and 
        //its log-probability score, and the content for each test. 
        //Insert a blank line after each for readability.
    // MODIFIES: -
    pair<int,int> test_classifier(char *argv[]) {
        string test_file = argv[2];
        csvstream csvin(test_file);
 
        int num_correct = 0;
        int num_posts = 0;

        cout << "test data:" << endl;
        pair<string,double> label_score;
        while(csvin >> correct_post) {
            label_score = predict_label();
            cout << "  correct = " << correct_post["tag"] << ", predicted = " <<
                label_score.first << ", log-probability score = " <<
                label_score.second << endl;
            cout << "  content = " << correct_post["content"] << "\n" << endl;

            if(correct_post["tag"] == label_score.first) {
                num_correct++;
            }
            num_posts++;
        }
        return {num_correct,num_posts};
    }

};

void check_command_line(int argc,char *argv[]) {
    bool correct_argc = false;
    if(argc == 3 || argc == 4) {correct_argc = true;}
    bool correct_debug = true;
    if (argc == 4) {
        string debug = argv[3];
        if(debug != "--debug") {
            correct_debug = false;
        }
    }

    if(!correct_argc || !correct_debug) {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
    }
}

int main(int argc, char *argv[]) {
    cout.precision(3);
    Classifier classifier;

    check_command_line(argc,argv);

    string train_file = argv[1];
    string test_file = argv[2];
    ifstream fin(train_file);
    ifstream fin2(test_file);
    if(!fin.is_open()) {
        cout << "Error opening file: " << train_file << endl;
        return 1;
    }
    if(!fin2.is_open()) {
        cout << "Error opening file: " << test_file << endl;
        return 1;
    }

    if(argc == 4) {
        string debug = argv[3];
        if(debug == "--debug") {
            classifier.print_label_content(argc,argv);
        }
    }

    classifier.train_classifier(argc,argv);

    cout << "trained on " << classifier.get_total_posts() << " examples" << endl;

    if(argc == 4) {
        string debug = argv[3];
        if(debug == "--debug") {
            cout << "vocabulary size = " << classifier.get_vocab_size() << endl;
        }
    }

    cout << "\n";

    if(argc == 4) {
        string debug = argv[3];
        if(debug == "--debug") {
            classifier.print_debug_data(argc,argv);
        }
    }

    pair<int,int> result = classifier.test_classifier(argv);

    cout << "performance: " << result.first << " / " 
    << result.second << " posts predicted correctly";

    cout << "\n";

    return 0;
}
 