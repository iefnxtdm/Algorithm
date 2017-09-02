/// @file parser.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_PARSER_H
#define GOODCODER_SRC_PARSER_H

#include<vector>
#include<memory>
#include<com_log.h>
#include"string_util.h"
#include "parse_struct.h"
#include "type_factory.h"
namespace goodcoder{

class Parser{
public:
    Parser(){
    }
    ~Parser(){
        for (auto it : _store){
            delete it;
        }
        _store.clear();
    }

    int init(const std::string& text);
    int parse(const std::string &line);
    int judge_int(const std::string& str);
    int judge_str(const std::string& str);
    int judge_float(const std::string& str);
    int judge_array(const std::string& str, Form& f, int& len);
    int print();
    void clear();
    const std::vector<MyType*>& get_data() const{
        return _store;
    }
    //TODO
    int judge_user(const std::string& str);
private:
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
    TypeFactory _factory;
    std::vector<MyType*> _store;
};

}

#endif
