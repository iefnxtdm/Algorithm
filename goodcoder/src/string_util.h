/// @file string_util.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31

#ifndef GOODCODER_SRC_STRING_UTIL_H
#define GOODCODER_SRC_STRING_UTIL_H

#include<vector>
#include<string>
#include<sstream>
#include<com_log.h>
typedef std::uint64_t hash_t;
constexpr hash_t prime = 0x100000001B3ull;  
constexpr hash_t basis = 0xCBF29CE484222325ull;
//字符串hash
constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis)  {  
    return *str ? hash_compile_time(str+1, (*str ^ last_value) * prime) : last_value;  
}

constexpr unsigned long long operator "" _hash(char const* p, size_t) {  
    return hash_compile_time(p);  
} 

class StringUtil{
public:

    /// @brief SplitString �ָ��ַ�
    ///
    /// @param src
    /// @param delimiter
    /// @param fields
    /// @param field_num
    ///
    /// @return
    static size_t splits_tring(const std::string& src,
                              const std::string& delimiter,
                              std::vector<std::string>* fields,
                              int field_num) {
        fields->clear();
        size_t previous_pos = 0;
        size_t current_pos = 0;
        while (static_cast<int>(fields->size()) < field_num &&
                (current_pos = src.find(delimiter, previous_pos)) != std::string::npos) {
            fields->push_back(src.substr(previous_pos, current_pos - previous_pos));
            previous_pos = current_pos + delimiter.length();
        }

        // Add the last string
        // If the last string is delimiter, add emyty string to fields too.
        if (previous_pos <= src.length()) {
            fields->push_back(src.substr(previous_pos));
        }

        return fields->size();
    }

    /// @brief str_to_int 
    ///
    /// @param str
    ///
    /// @return
    static int str_to_int(const std::string& str, int &num){
        if (str.size() == 0){
            return -1;
        }
        std::stringstream ss;
        ss<<str;
        ss>>num;
        if (num == 2147483647 || num == -2147483648){
            com_writelog(COMLOG_WARNING, "str=%s, num overload, output INTMAX", str.c_str());
        }
        return 0;
    }

    /// @brief str_to_float ת��Ϊfloat
    ///
    /// @param str
    ///
    /// @return
    static int str_to_float(const std::string& str, float &num){
        if (str.size() == 0){
            return -1;
        }
        std::stringstream ss;
        ss<<str;
        ss>>num;
        return 0;
    }
};

#endif
