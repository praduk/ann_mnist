/*
* ann.cc
*
* Feedforward Fully-Connected Neural Network Architecture
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#include "ann.hh"

#include <stdio.h>
#include <stdlib.h>

//double uniform random number
double drand(double fMin, double fMax)
{
    double f = ((double)(rand()))/RAND_MAX;
    return fMin + f*(fMax-fMin);
}

Layer::Layer(int nI, int nO) : I(nI), O(nO)
{
    if(O > 0)
        o = new double[O];
    else
        o = 0;
    if(O > 0)
        g = new double[I];
    else
        g = 0;
}

Layer::~Layer()
{
    if( o )
        delete [] o;
    if( g )
        delete [] g;
}

void SoftMax::eval()
{
    sum = 0.0;
    for( int i=0; i<I; i++ )
    {
        sum += in[i];
        o[i] = in[i];
    }
    if( sum!=0.0 )
        for(int i=0; i<I; i++)
            o[i] /= sum;
    else
        for(int i=0; i<I; i++)
            o[i] = 0.0;
}

void SoftMax::backprop()
{
    double invsum2 = 1.0/(sum*sum);
    double C = 0.0;
    for( int j=0; j<I; j++ )
        C -= in[j]*og[j]*invsum2;
    for( int i=0; i<I; i++ )
        g[i] = C + og[i]*sum*invsum2;
}

LinearLayer::LinearLayer(int nI, int nO)
    : Layer(nI, nO), IO(nI*nO)
{
    w = new double[IO];
    b = new double[O];
    z = new double[O];

    //initialize parameters
    for( int i=0; i<IO; i++ )
        w[i] = drand(-1.0,1.0);
    for( int i=0; i<O; i++ )
        b[i] = drand(-1.0,1.0);

    // parameter gradients
    wg = new double[IO];
    bg = new double[O];
    clear();
}

LinearLayer::~LinearLayer()
{
    delete [] w;
    delete [] b;
    delete [] z;
    delete [] wg;
    delete [] bg;
}

void LinearLayer::clear()
{
    for(int i=0; i<IO; i++)
        wg[i] = 0.0;
    for(int i=0; i<O; i++)
        bg[i] = 0.0;
}

void LinearLayer::step(double h)
{
    //printf("Stepping Linear Layer\n");
    for(int i=0; i<IO; i++)
    {
        w[i] -= wg[i]*h;
    }
    for(int i=0; i<O; i++)
    {
        //printf("b[%d] %23.15E - %23.15E\n",i,b[i],bg[i]*h);
        b[i] -= bg[i]*h;
    }
}

void LinearLayer::eval()
{
    for(int r=0; r<O; r++)
        o[r] = 0.0;
    for(int r=0; r<O; r++)
        for(int c=0; c<I; c++)
            o[r] += W(r,c)*in[c];
    for(int r=0; r<O; r++)
    {
        z[r] = o[r] + b[r];
        o[r] = fsigma(z[r]);
    }
}

void LinearLayer::backprop()
{
    //reset input gradients
    for(int c=0; c<I; c++)
        g[c] = 0.0;
    for(int r=0; r<O; r++)
    {
        double K = og[r]*dsigma(z[r]);

        //accumulate bias gradients
        bg[r] += K;

        for(int c=0; c<I; c++)
        {
            //accumulate weight gradients
            WG(r,c) += K*in[c];

            //accumulate input gradients
            g[c] += K*W(r,c);
        }
    }
}

double LinearLayer::normgrad2()
{
    double sum = 0.0;
    for(int i=0; i<O; i++)
        sum += bg[i]*bg[i];
    for( int i=0; i<IO; i++ )
        sum += wg[i]*wg[i];
    return sum;
}

void Network::addLayer(Layer* l)
{
    if( layers.size() > 0 )
        l->setInput(layers.back());
    layers.push_back(l);
}

void Network::eval()
{
    layers.front()->setInput(in);
    for( std::vector<Layer*>::size_type i=0; i<layers.size(); i++)
        layers[i]->eval();
    for( int i=0; i<O; i++ )
        o[i] = layers.back()->o[i];
}

void Network::backprop()
{
    layers.back()->setGrad(og);
    for(int i=layers.size()-1; i>=0; i--)
        layers[i]->backprop();
    for( int i=0; i<I; i++ )
        g[i] = layers.front()->g[i];
}

void Network::clear()
{
    for( std::vector<Layer*>::size_type i=0; i<layers.size(); i++)
        layers[i]->clear();
}

void Network::step(double h)
{
    for( std::vector<Layer*>::size_type i=0; i<layers.size(); i++)
        layers[i]->step(h);
}

double Network::normgrad2()
{
    double sum = 0.0;
    for( std::vector<Layer*>::size_type i=0; i<layers.size(); i++)
        sum += layers[i]->normgrad2();
    return sum;
}
