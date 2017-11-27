/*
* ann.hh
*
* Feedforward Fully-Connected Neural Network Architecture
* 
* Copyright (c) 2017 Pradu Kannan. All rights reserved.
*/

#ifndef ___ANN_HH
#define ___ANN_HH

#include <cmath>
#include <vector>

//M inputs per node, N outputs per node
struct Layer
{
    int I; //num inputs
    int O; //num outputs

    //Layer Inputs
    double const* in; //input vector
    double const* og; //backprop gradient vector

    //Layer States
    double* o; //output
    double* g; //gradients of input vector

    Layer(int nI, int nO);
    virtual ~Layer();

    //set pointer to input vector
    void setInput(double const* in_in) { in = in_in; }

    //set pointer to backprop gradient vector
    void setGrad(double const* og_in) { og = og_in; }

    void setInput(Layer* l) { setInput(*l); }
    void setInput(Layer& l)
    {
        setInput(l.o);
        l.setGrad(g);
    }

    //evaluate network - uses pointer to last input reference
    virtual void eval() = 0;

    //backprop gradients
    virtual void backprop() = 0;

    //clear accumulated gradients of layer parameters
    virtual void clear() {};

    //step layer parameters using accumulated gradients
    virtual void step(double h) {};

    //return norm of parameter gradient squared
    virtual double normgrad2() { return 0.0; }
};

struct SoftMax : public Layer
{
    double sum;

    SoftMax(int nI) : Layer(nI,nI) {}

    void eval();
    void backprop();
};

struct LinearLayer : public Layer
{
    int IO; //product of I and O

    double* z;  //linear output
    double* w;  //weights
    double* wg; //weight gradients
    double* b;  //bias
    double* bg; //bias gradients

    LinearLayer(int nI, int nO);
    ~LinearLayer();

    //get index i,j of weight matrix
    double& W(int r, int c) { return *(w + I*r + c); }

    //get index i,j of weight gradient
    double& WG(int r, int c) { return *(wg + I*r + c); }

    //Output Wrapper Function and Derivative
    virtual double fsigma(double x) { return x; }
    virtual double dsigma(double x) { return 1.0; }

    void eval();
    void backprop();
    void clear();
    void step(double h);
    double normgrad2();
};

struct LogisticLayer : public LinearLayer
{
    LogisticLayer(int nI, int nO) : LinearLayer(nI, nO) {}
    double fsigma(double x) { return 1.0/(1.0+std::exp(-x)); }
    double dsigma(double x)
    {
        double emx = std::exp(-x);
        double emxp1inv = 1.0/(1.0+emx);
        return emx*emxp1inv*emxp1inv;
    }
};

struct SoftPlusLayer : public LinearLayer
{
    SoftPlusLayer(int nI, int nO) : LinearLayer(nI, nO) {}
    double fsigma(double x) { std::log(1.0+exp(x)); }
    double dsigma(double x) { return 1.0/(1.0+std::exp(-x)); }
};

struct RELULayer : public LinearLayer
{
    RELULayer(int nI, int nO) : LinearLayer(nI, nO) {}
    double fsigma(double x)
    {
        if( x<=0.0 )
            return 0.0;
        else
            return x;
    }
    double dsigma(double x)
    {
        if( x<=0.0 )
            return 0.0;
        else
            return 1.0;
    }
};

struct Network : public Layer
{
    std::vector<Layer*> layers;
    Network(int nI, int nO) : Layer(nI,nO) {}

    void addLayer(Layer* l);
    void eval();
    void backprop();
    void clear();
    void step(double h);
    double normgrad2();
};

#endif  // ___ANN_HH

