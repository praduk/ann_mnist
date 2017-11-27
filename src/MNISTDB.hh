/*
* MNISTDB.hh
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#ifndef ___MNISTDB_HH
#define ___MNISTDB_HH

typedef unsigned int U32;
typedef unsigned char U8;

#include <vector>

struct MNISTimg
{
    enum { DATASZ=784 };
    U8 data[28][28];
    U8 label;
};

class MNISTDB : public std::vector<MNISTimg>
{
public:
    MNISTDB(char const* fn_images, char const* fn_labels);
};

#endif  // ___MNISTDB_HH

