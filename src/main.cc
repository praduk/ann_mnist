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

//void checkBackprop(Layer& l)
//{
//    double h = 1E-9; //step size
//    double* in = new double[l.I];
//    double* og = new double[l.O];
//    l.setInput(in);
//    l.setGrad(og);
//    for(int i=0; i<l.I; i++)
//        in[i] = drand(-1.0,1.0);
//    for(int i=0; i<l.O; i++)
//        og[i] = drand(-1.0,1.0);
//
//    l.clear();
//    l.eval();
//    l.backprop();
//
//    double J0 = 0.0;
//    for( int i=0; i<l.O; i++)
//        J0 += og[i]*l.o[i];
//
//    //perturb each input and evaluate gradient
//    for(int i=0; i<l.I;i++)
//    {
//        double inorig = in[i];
//        in[i] += h;
//
//        l.eval();
//        double J = 0.0;
//        for( int j=0; j<l.O; j++)
//            J += og[j]*l.o[j];
//
//        printf("%02d %23.15E %23.15E\n", i, (J-J0)/h, l.g[i]);
//
//        in[i] = inorig;
//    }
//
//    delete [] in;
//    delete [] og;
//}

//void checkBackprop(LinearLayer& l)
//{
//    double h = 1E-9; //step size
//    double* in = new double[l.I];
//    double* og = new double[l.O];
//    l.setInput(in);
//    l.setGrad(og);
//    for(int i=0; i<l.I; i++)
//        in[i] = drand(-1.0,1.0);
//    for(int i=0; i<l.O; i++)
//        og[i] = drand(-1.0,1.0);
//
//    l.clear();
//    l.eval();
//    l.backprop();
//
//    double J0 = 0.0;
//    for( int i=0; i<l.O; i++)
//        J0 += og[i]*l.o[i];
//
//    //perturb each input and evaluate gradient
//    for(int i=0; i<l.I;i++)
//    {
//        double inorig = in[i];
//        in[i] += h;
//
//        l.eval();
//        double J = 0.0;
//        for( int j=0; j<l.O; j++)
//            J += og[j]*l.o[j];
//
//        printf("I %02d %23.15E %23.15E\n", i, (J-J0)/h, l.g[i]);
//
//        in[i] = inorig;
//    }
//
//    //perturb each bias and evaluate gradient
//    for(int i=0; i<l.O;i++)
//    {
//        double borig = l.b[i];
//        l.b[i] += h;
//
//        l.eval();
//        double J = 0.0;
//        for( int j=0; j<l.O; j++)
//            J += og[j]*l.o[j];
//
//        printf("B %02d %23.15E %23.15E\n", i, (J-J0)/h, l.bg[i]);
//
//        l.b[i] = borig;
//    }
//
//    //perturb each weight and evaluate gradient
//    for(int i=0; i<l.IO;i++)
//    {
//        double worig = l.w[i];
//        l.w[i] += h;
//
//        l.eval();
//        double J = 0.0;
//        for( int j=0; j<l.O; j++)
//            J += og[j]*l.o[j];
//
//        printf("W %02d %23.15E %23.15E\n", i, (J-J0)/h, l.wg[i]);
//
//        l.w[i] = worig;
//    }
//
//    delete [] in;
//    delete [] og;
//}

void train(Network& n, MNISTDB& db)
{
    const double gamma = 5E-1; //learning rate
    const double scale = 1.0/255.0; //Scale Pixel Values from 0-255 to 0-1
    double input[MNISTimg::DATASZ]; //Input Vector
    double error[10]; //Eroror Vector, output - desired
    int dsize = 1;

    n.setInput(input);
    n.setGrad(error);

    double h = gamma;
    for( int it=0; it<10000; it++)
    {
        //Train
        n.clear(); //Clear Gradients
        //Calculate Gradients
        double J = 0.0; //cost function
        //for(int i=0; i<db.size(); i++)
        int osize = db.size()/dsize;
        int part = rand()%dsize;
        
        //for(int i=0; i<db.size(); i++)
        for(int i=osize*part; i<osize*part+osize; i++)
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
        n.step(h);
        //Calculate New Cost
        double Jnew = 0.0;
        //for(int i=0; i<db.size(); i++)
        for(int i=osize*part; i<osize*part+osize; i++)
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

        double expDel = -h*n.normgrad2();
        double actDel = Jnew-J;
        double ratio = actDel/expDel;

        if( ratio < 0.5 )
        {
            n.step(-h);
            h /= 2.0;
        }
        else if( fabs(ratio-1.0) <= 0.05 )
            h *=1.1;
        else
            h /= 1.09;
        //else if ( ratio <= 0.5 )
        //    h /=1.2;
        
        //Print Statistics
        printf("%05d J=%23.15E act/exp=%23.15E h=%23.15E\n",it,J*dsize,ratio,h);
        if( it%(10*dsize)==0 )
        {
            printf("Saving Weights\n");
            FILE* f = fopen("weights.bin","wb");
            for( int i=0; i<(n.layers.size()-1); i++ )
            {
                LinearLayer* p = static_cast<LinearLayer*>(n.layers[i]);
                fwrite(p->w, sizeof(double), p->IO, f);
                fwrite(p->b, sizeof(double), p->O, f);
            }
            fclose(f);
        }
    }
}

void test(Network& n, MNISTDB& db)
{
    double input[MNISTimg::DATASZ]; //Input Vector
    const double scale = 1.0/255.0; //Scale Pixel Values from 0-255 to 0-1
    int correct = 0;
    n.setInput(input);
    for( int i=0; i<db.size(); i++)
    {
        MNISTimg& sample = db[i];
        for(int j=0; j<MNISTimg::DATASZ; j++)
            input[j] = scale*(*(U8*)(sample.data+j));
        n.eval();
        int maxidx = 0;
        for( int j=1; j<10; j++)
            if( n.o[j] > n.o[j-1] )
                maxidx = j;
        if( maxidx == sample.label )
            correct++;
    }
    printf("%f percent correct\n", 100*double(correct)/db.size());
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
    //lin.addLayer(new LinearLayer(MNISTimg::DATASZ,10));
    lin.addLayer(new LogisticLayer(MNISTimg::DATASZ,16));
    lin.addLayer(new LogisticLayer(16,10));
    lin.addLayer(new SoftMax(10));

    {
        FILE* f = fopen("weights.bin","rb");
        if( f!= NULL )
        {
            for( int i=0; i<(lin.layers.size()-1); i++ )
            {
                LinearLayer* p = static_cast<LinearLayer*>(lin.layers[i]);
                fread(p->w, sizeof(double), p->IO, f);
                fread(p->b, sizeof(double), p->O, f);
            }
            fclose(f);
        }
    }
    train(lin,traindb);
    //test(lin,testdb);


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
    
    //LinearLayer ll(3,2);
    //checkBackprop(ll);
    //SoftMax ll(10);
    //SoftPlusLayer ll(16,10);
    //LogisticLayer ll(16,10);
    //RELULayer ll(16,10);
    //checkBackprop(lin);

    return 0;
}
