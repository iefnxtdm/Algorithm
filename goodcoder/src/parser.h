/// @file parser.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_PARSER_H
#define GOODCODER_SRC_PARSER_H
#include<vector>
using std::vector;
namespace goodcoder{

class Parser{
public:
    Parser(){
        _store = new vector<Type*>();  
    }
    ~Parser(){
        for(auto it:_store){
            delete it;
        }
        _store.clear();
        delete _store;
    }
    bool parse(const std::string& line);
    bool judge_int(const std::string& str);
    bool judge_str(const std::string& str);
    bool judge_float(const std::string& str);
    bool judge_array(const std::string& str);
    //TODO
    bool judge_user(const std::string& str);
private:
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
    vector<Type*> *_store;
};

}

#endif
