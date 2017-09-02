/// @file format_parser.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-8-10
#ifndef GOODCODER_SRC_FORMAT_PARSER_H
#define GOODCODER_SRC_FORMAT_PARSER_H

#include "parser.h"
#include <utility>
namespace goodcoder{

class FormatParser : public Parser{
public:
    FormatParser(){
    }
    ~FormatParser(){
        for (auto it : _store){
            delete it;
        }
        _store.clear();
        _type_v.clear();
    }

    int init(const std::string& text);
    int parse(const std::string &line);
    Form judge_array_type(const std::string& str, int& len);
    int print();
    void clear();
    const std::vector<MyType*>& get_data() const{
        return _store;
    }
private:
    FormatParser(const Parser&) = delete;
    FormatParser& operator=(const Parser&) = delete;
    TypeFactory _factory;
    std::vector<MyType*> _store;
    std::vector<std::pair<Form, int>> _type_v;
};

}

#endif
