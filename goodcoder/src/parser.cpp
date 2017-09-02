/// @file parser.cpp
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#include<vector>
#include<string>
#include <fstream>
#include"parser.h"
#include"type_factory.h"
using std::vector;
using std::string;
namespace goodcoder{

    int Parser::init(const string& path){
        string line;
        std::ifstream in(path.c_str());
        while (getline(in, line)){
            clear();
            com_writelog(COMLOG_NOTICE, "source line = [%s]", line.c_str());
            if (parse(line) != 0){
                com_writelog(COMLOG_WARNING, "%s parse err", line.c_str());
                continue;
            }
            print();
        }
        in.close();
    }
    int Parser::print(){
        for (auto t : _store){
            t->print();
        }
        return 0;
    }
    void Parser::clear(){
        for (size_t i = 0; i < _store.size(); ++i){
            delete _store[i];
        }
        _store.clear();
    }
    int Parser::parse(const std::string& line){
        vector<string> res;
        StringUtil::splits_tring(line, "\t", &res, 100);
        for (size_t i = 0; i < res.size(); ++i){
            const string& tmp = res[i];
            Form f;
            int len = 0;
            MyType * mtp = nullptr;
            if (!judge_array(tmp, f, len)){
                mtp = _factory.create_type(f, tmp, len);
                com_writelog(COMLOG_DEBUG, "array_len=%d", len);
                _store.push_back(mtp);
                continue;
            }else if (!judge_float(tmp)){
                mtp = _factory.create_type(Form::FLOAT, tmp, 1);
                _store.push_back(mtp);
                continue;
            }else if (!judge_int(tmp)){
                mtp = _factory.create_type(Form::INT, tmp, 1);
                _store.push_back(mtp);
                continue;
            }else if (!judge_user(tmp)){
                mtp = _factory.create_type(Form::USER, tmp, 1);
                _store.push_back(mtp);
                continue;
            }else if (!judge_str(tmp)){
                mtp = _factory.create_type(Form::STRING, tmp, tmp.size());
                _store.push_back(mtp);
                continue;
            }
        }
        return 0;
    }
    int Parser::judge_int(const std::string& str){
        bool flag = true;
        if (str.size() == 0){
            return -1;
        }
        for (size_t i = 0; i < str.size(); ++i){
            if (i == 0 && (str[i] == '+' || str[i] == '-')){
                continue;
            }
            if (str[i] >= '0' && str[i] <= '9'){
                continue;
            }
            flag = false;
            break;
        }
        return flag? 0: -1;
    }
    int Parser::judge_str(const std::string& str) {
        return 0;
    }
    int Parser::judge_float(const std::string& str){
        std::string::size_type pos1 = str.find_first_of(".");
        std::string::size_type pos2 = str.find_last_of(".");
        if (pos1 == std::string::npos || pos1 != pos2 ||
            pos1 == 0 || pos1 == str.size() - 1){
            return -1;
        }
        bool flag = true;
        for (size_t i = 0; i < str.size(); ++i){
            if (i == 0 && (str[i] == '+' || str[i] == '-')){ //判断符号
                continue;
            }
            if ((str[i] >= '0' && str[i]<= '9') || str[i] == '.'){
                continue;
            }
            flag = false;
            break;
        }
        return flag? 0: -1;
    }
    int Parser::judge_array(const std::string& str, Form& f, int& len){
        auto pos1 = str.find_first_of(":");
        auto pos2 = str.find_last_of(":");
        if (pos1 == std::string::npos || pos1 != pos2){
            return -1;
        }
        auto len_str = str.substr(0, pos1);
        if (judge_int(len_str) == 0){
            StringUtil::str_to_int(len_str, len);
        }else{
            len = 0;
            com_writelog(COMLOG_WARNING, "array len=[%d] error", len);
            return -1;
        }
        vector<string> res;
        auto arr_str = str.substr(pos1 + 1);
        StringUtil::splits_tring(arr_str, ",", &res, 100);
        int flag = 0; // 1:int 2:float
        for (int i = 0; i < res.size(); i++){
            if (i == 0){
                if (!judge_float(res[i])){
                    flag = 2;
                }else if (!judge_int(res[i])){
                    flag = 1;
                }
                else{
                    com_writelog(COMLOG_WARNING, "array[0]=[%s] format err, len=[%d]",
                        res[i].c_str(), len);
                    return -1;
                }
            }
            if (flag == 2 && !judge_float(res[i])){
                f = FLOAT_ARRAY;
                continue;
            }
            if (flag == 1 && !judge_int(res[i])){
                f = INT_ARRAY;
                continue;
            }
            com_writelog(COMLOG_WARNING, "a[%d]=[%s] format err, arr_format=[%d]", i, res[i].c_str(), f);
            f = NONE;
            return -1;
        }
        return 0;
    }
    //TODO
    int Parser::judge_user(const std::string& str){
        size_t pos = str.find("-");
        if (pos == std::string::npos){
            return -1;
        }
        if (!judge_int(str.substr(0, pos)) && !judge_int(str.substr(pos+1))){
            return 0;
        }
        return -1;
    }

}
