/// @file format_parser.cpp
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-08-10
#include<vector>
#include<string>
#include <fstream>
#include <string>
#include"format_parser.h"
#include"type_factory.h"
using std::vector;
using std::string;
namespace goodcoder{

    int FormatParser::print(){
        for (auto t : _store){
            t->print();
        }
        return 0;
    }
    void FormatParser::clear(){
        for (size_t i = 0; i < _store.size(); ++i){
            delete _store[i];
        }
        _store.clear();
    }
    int FormatParser::init(const string& text){
        vector<string> res;
        std::fstream in(text.c_str());
        if (in.is_open()) {
            string line;
            getline(in, line);
            com_writelog(COMLOG_NOTICE, "format type=[%s]", line.c_str());
            StringUtil::splits_tring(line, "\t", &res, 100);
            for (size_t i = 0; i < res.size(); ++i){
                switch (hash_compile_time(res[i].c_str())){
                    case "int"_hash:
                        _type_v.push_back(std::make_pair(Form::INT, 1));
                        break;
                    case "float"_hash:
                        _type_v.push_back(std::make_pair(Form::FLOAT, 1));
                        break;
                    case "user"_hash:
                        _type_v.push_back(std::make_pair(Form::USER, 1));
                        break;
                    case "string"_hash:
                        _type_v.push_back(std::make_pair(Form::STRING, 1));
                        break;
                    default: //解析是否数组
                        int len = 0;
                        Form ret = judge_array_type(res[i], len);
                        if (ret != -1){
                            _type_v.push_back(std::make_pair(ret, len));
                        } else {
                            _type_v.push_back(std::make_pair(Form::STRING, 1));
                        }
                        break;
                }
            }
            in.close();
            return 0;
        }
        com_writelog(COMLOG_WARNING, "format_parser open file error!");
        return -1;
    }

    Form FormatParser::judge_array_type(const std::string& str, int& len){
        size_t pos1 = str.find("int");
        size_t pos2 = str.find("float");
        size_t start = 0;
        Form type = NONE;
        if (pos1 != 0 && pos2 != 0){
            return NONE;
        } else if (pos1 == 0){
            start = 4; //int 开始
            type = Form::INT_ARRAY;
        } else {
            start = 6;
            type = Form::FLOAT_ARRAY;
        }
        if (str.size() <= start){
            return NONE;
        }
        len = std::stoi(str.substr(start), nullptr); //TODO
        com_writelog(COMLOG_DEBUG, "array_type type=%d, len=%d", type, len);
        return type;
    }

    int FormatParser::parse(const std::string& path){
        std::fstream in(path.c_str());
        if (!in.is_open()){
            return -1;
        }
        std::string line;
        std::vector<string> res;
        int first_flag = true;
        while (getline(in, line)){
            clear();
            if (first_flag){ //去掉首行配置
                first_flag = false;
                continue;
            }
            com_writelog(COMLOG_NOTICE, "source line=[%s]", line.c_str());
            StringUtil::splits_tring(line, "\t", &res, 100);
            if (res.size() != _type_v.size()){
                com_writelog(COMLOG_WARNING, "type&&data column dont match");
                return -1;
            }
            int judge_flag = 0;
            for (size_t i = 0; i < res.size(); ++i){
                //judge_flag = judge_format_type(_type_c[i].first, _type_c[i].second, res[i]);
                //TOTO 容错判断
                MyType* mtp = nullptr;
                mtp = _factory.create_type(_type_v[i].first, res[i], _type_v[i].second);
                _store.push_back(mtp); 
            }
            print();
        }
        in.close();
        return 0;   
    }
}