/// @file parser.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_PARSER_H
#define GOODCODER_SRC_PARSER_H

#include<vector>
#include<memory>
#include"string_util.h"
#include"type_spec.h"
using std::vector;
namespace goodcoder{

class Parser{
public:
    Parser(){
        _store = make_shared(new vector<Type*>());  
    }
    ~Parser(){
        for(auto it:_store){
            delete it;
        }
        _store->clear();
    }
    int parse(const std::string& line);
    int judge_int(const std::string& str, Type* store);
    int judge_str(const std::string& str, Type* store);
    int judge_float(const std::string& str, Type* store);
    int judge_array(const std::string& str, Type* store);
    //TODO
    int judge_user(const std::string& str, Type* store);
private:
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
    std::shared_ptr<std::vector<Type*>> _store;
};

}

#endif
