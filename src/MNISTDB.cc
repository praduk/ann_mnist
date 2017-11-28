/*
* MNISTDB.cc
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#include "MNISTDB.hh"
#include "config.hh"
#include <stdio.h>

#ifndef U_BIG_ENDIAN
U32 bswap(U32 x)
{
    return (x>>24) | (x<<24) |
           ((x<<8) & 0x00FF0000) |
           ((x>>8) & 0x0000FF00);
}
#else
#define bswap(x) x
#endif

void MNISTimg::print()
{
    for(int r=0; r<28;r++)
    {
        for(int c=0; c<28; c++)
        {
            printf("%c[48;2;%d;%d;%dm  ", 0x1B, data[r][c], data[r][c], data[r][c]);
        }
        printf("%c[0m\n",0x1B);
    }
    printf("        Label = %d\n",label);

}

MNISTDB::MNISTDB(char const* fn_images, char const* fn_labels)
{
    FILE* fimg = fopen(fn_images,"rb");
    if( fimg == NULL )
        return;
    FILE* flbl = fopen(fn_labels,"rb");
    if( flbl == NULL )
    {
        fclose(fimg);
        return;
    }

    //Read Magic
    {
        U32 magic;
        fread(&magic,sizeof(magic),1,fimg);
        if(bswap(magic)!=2051)
        {
            printf("%u\n",bswap(magic));
            printf("%u\n",magic);
            printf("Program Compiled for Wrong Endianness\n");
            fclose(flbl); fclose(fimg); return;
        }
        fread(&magic,sizeof(magic),1,flbl);
        if(bswap(magic)!=2049)
        {
            printf("Program Compiled for Wrong Endianness\n");
            fclose(flbl); fclose(fimg); return;
        }
    }

    U32 numItems;
    fread(&numItems,sizeof(numItems),1,fimg);
    numItems = bswap(numItems);
    {
        U32 numItems2;
        fread(&numItems2,sizeof(numItems2),1,flbl);
        numItems2 = bswap(numItems2);
        if( numItems2 != numItems )
        {
            printf("Number of Items do not Match\n");
            fclose(flbl); fclose(fimg); return;
        }
    }

    //read rows and cols (assuemd 28)
    {
        U32 x;
        fread(&x,sizeof(x),1,fimg); //rows
        fread(&x,sizeof(x),1,fimg); //cols
    }

    MNISTimg img;
    for(U32 i=0; i<numItems; i++)
    {
        fread(&img.data,1,MNISTimg::DATASZ,fimg);
        fread(&img.label,1,1,flbl);
        push_back(img);
    }

    fclose(flbl);
    fclose(fimg);

    printf("Read MNIST Dataset with %u images\n", size());
    printf("    Images File: %s\n", fn_images);
    printf("    Labels File: %s\n", fn_labels);
}
