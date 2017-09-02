/// @file type_int.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_SPEC_H
#define GOODCODER_SRC_TYPE_SPEC_H

#include<iostream>
#include <string>
#include <vector>
#include <sstream>
#include <com_log.h>
#include "mytype.h"
#include "string_util.h"

namespace goodcoder{
class TypeInt : public MyType{
public:
    TypeInt(const std::string& str, int len){
        _str_type = Form::INT;
        _len = len;
        _var = new int[len];
        set_val(str);
    }
    ~TypeInt(){
        delete _var;
        _var = nullptr;
    }
    int set_val(const std::string& str){
        std::vector<std::string> tmp;
        int res = 0;
        size_t loc = str.find(":");
        std::string for_split;
        if (loc != std::string::npos){ //是数组
            for_split = str.substr(loc + 1);
        } else {
            for_split = str;
        }
        com_writelog(COMLOG_DEBUG, "for_split=%s", for_split.c_str());
        StringUtil::splits_tring(for_split, ",", &tmp, 100);
        for (size_t i = 0; i < tmp.size(); ++i){
            StringUtil::str_to_int(tmp[i], res);
            _var[i] = res;
            com_writelog(COMLOG_DEBUG, "str=[%s], res=[%d]", tmp[i].c_str(), res);
        }
        return 0;
    }
    int get_val(void*& p, int& len, Form& form){
        p = _var;
        len = _len;
        form = Form::INT;
        return 0;
    }
    void print(){
        std::stringstream ss;
        ss << "var=int;len=" << _len << " | ";
        for (int i = 0; i < _len; ++i){
            ss << _var[i] << " ";
        }
        com_writelog(COMLOG_NOTICE, "%s", ss.str().c_str());
    }
private:
    int* _var;
    int _len;
    Form _str_type;
};

class TypeFloat : public MyType{
public:
    TypeFloat(const std::string& str, int len){
        _str_type = Form::FLOAT;
        _len = len;
        _var = new float[_len];
        set_val(str);
    }
    ~TypeFloat(){
        delete _var;
        _var = nullptr;
    }
    int set_val(const std::string& str){
        std::vector<std::string> tmp;
        float res = 0;
        size_t loc = str.find(":");
        std::string for_split;
        if (loc != std::string::npos){ //是数组
            for_split = str.substr(loc + 1);
        } else {
            for_split = str;
        }
        StringUtil::splits_tring(for_split, ",", &tmp, 100);
        for (size_t i = 0; i < tmp.size(); ++i){
            StringUtil::str_to_float(tmp[i], res);
            _var[i] = res;
        }
        return 0;
    }
    int get_val(void*& p, int &len, Form& form){
        p = _var;
        len = _len;
        form = Form::FLOAT;
        return 0;
    }
    void print(){
        std::stringstream ss;
        ss << "var=float;len=" << _len << " | ";
        for (int i = 0; i < _len; ++i){
            ss << _var[i] << " ";
        }
        com_writelog(COMLOG_NOTICE, "%s", ss.str().c_str());
    }
private:
    float* _var;
    int _len;
    Form _str_type;
};

class TypeStr : public MyType{
public:
    TypeStr(const std::string& str, int len){
        _str_type = Form::STRING;
        _len = len;
        _var = str;
        set_val(str);
    }
    void print(){
        com_writelog(COMLOG_NOTICE, "var=str | %s", _var.c_str());
    }
    int set_val(const std::string& str){
        _var = str;
        return 0;
    }
    int get_val(void*& p, int &len, Form& form){
        p = &_var;
        len = _len;
        form = Form::STRING;
        return 0;
    }
private:
    std::string _var;
    int _len;
    Form _str_type;
};

class TypeUser : public MyType{
public:
    TypeUser(const std::string& str){
        _str_type = Form::USER;
        _t = new int[2];
        set_val(str);
    }
    ~TypeUser(){
        delete _t;
        _t = nullptr;
    }
    void print(){
        com_writelog(COMLOG_NOTICE, "var=user | %d-%d", _t[0], _t[1]);
        return;
    }
    int set_val(const std::string& str){
        size_t pos = str.find("-");
        StringUtil::str_to_int(str.substr(0, pos), _t[0]);
        StringUtil::str_to_int(str.substr(pos+1), _t[1]);
        return 0;
    }
    int get_val(void*& p, int &len, Form& form){
        p = _t;
        len = 2;
        form = Form::USER;
        return 0;
    }
private:
    int *_t;
    Form _str_type;
};
}
#endif
