/*
* main.cc
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#include "config.hh"
#include "MNISTDB.hh"
#include "ann.hh"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void train(Network& n, MNISTDB& db)
{
    const double gamma = 1E-15; //learning rate
    const double scale = 1.0/255.0; //Scale Pixel Values from 0-255 to 0-1
    double input[MNISTimg::DATASZ]; //Input Vector
    double error[10]; //Eroror Vector, output - desired

    n.setInput(input);
    n.setGrad(error);

    for( int it=0; it<1; it++)
    {
        //Train
        n.clear(); //Clear Gradients
        //Calculate Gradients
        double J = 0.0; //cost function
        for(int i=0; i<db.size(); i++)
        //for(int i=0; i<1; i++)
        {
            MNISTimg& sample = db[i];
            for(int j=0; j<MNISTimg::DATASZ; j++)
                input[j] = scale*(*(U8*)(sample.data+j));
            n.eval();
            for(int j=0; j<10; j++)
                error[j] = n.o[j];
            error[sample.label] -= 1.0;
            for(int j=0; j<10; j++)
                J += 0.5*error[j]*error[j];
            n.backprop();
        }
        //Perform Gradient Step
        n.step(gamma);
        //Calculate New Cost
        double Jnew = 0.0;
        for(int i=0; i<db.size(); i++)
        //for(int i=0; i<1; i++)
        {
            MNISTimg& sample = db[i];
            for(int j=0; j<MNISTimg::DATASZ; j++)
                input[j] = scale*(*(U8*)(sample.data+j));
            n.eval();
            for(int j=0; j<10; j++)
                error[j] = n.o[j];
            error[sample.label] -= 1.0;
            for(int j=0; j<10; j++)
                Jnew += 0.5*error[j]*error[j];
        }

        double expDel = -gamma*sqrt(n.normgrad2());
        double actDel = Jnew-J;
        
        //Print Statistics
        printf("  %03d J=%23.15E act/exp=%23.15E\n",it,J,actDel/expDel);
    }
}

int main()
{
    srand(0);
    //Training Dataset
    MNISTDB traindb(
        DATA_DIR "/train-images-idx3-ubyte",
        DATA_DIR "/train-labels-idx1-ubyte");

    //Test Dataset
    MNISTDB testdb(
        DATA_DIR "/t10k-images-idx3-ubyte",
        DATA_DIR "/t10k-labels-idx1-ubyte");

    //Train and Compare
    //(1) LinearLayer + SoftMax Network
    //(2) Logistic Network
    //(3) SoftPlus Network
    //(4) RELU Network

    
    //LinearLayer + SoftMax Network
    Network lin(MNISTimg::DATASZ,10);
    lin.addLayer(new LinearLayer(MNISTimg::DATASZ,10));
    //lin.addLayer(new LogisticLayer(MNISTimg::DATASZ,16));
    //lin.addLayer(new LogisticLayer(16,10));
    //lin.addLayer(new SoftMax(10));
    train(lin,traindb);

    //LinearLayer ll(1,1);
    //double input[2];
    //double grad[1];
    //ll.setInput(input);
    //ll.setGrad(grad);

    //for( int k=0; k<10; k++)
    //{
    //    ll.clear();

    //    //x=0, y=3
    //    input[0] = 0.0;
    //    ll.eval();
    //    grad[0] = ll.o[0]-3.0;
    //    ll.backprop();
    //    //x=1, y=3.5
    //    input[0] = 1.0;
    //    ll.eval();
    //    grad[0] = ll.o[0]-3.5;
    //    ll.backprop();


    //    for(int r=0; r<ll.O; r++)
    //        for(int c=0; c<ll.I; c++)
    //            printf("W(%d,%d) : %f %f\n",r,c,ll.W(r,c),ll.WG(r,c));
    //    for(int r=0; r<ll.O; r++)
    //        printf("b(%d) : %f %f\n",r,ll.b[r],ll.bg[r]);
    //    ll.step(0.4);
    //}

    return 0;
}
