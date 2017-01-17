#include "main.h"

using namespace std;

float** train_inputs;   // NVars x N_train
float** train_batch;    // NVars x mini_batch_size
float** validation_inputs;  // NVars x N_validation
float** test_inputs;    // NVars x N_test
int** train_targets;    // NClasses x N_train
int** train_batch_targets;  // NClasses x mini_batch_size
int** validation_targets;   // NClasses x N_validation
int** test_targets;     // NClasses x N_test

int N_train;
int N_validation;
int N_test;
int MaxSampleSize;

int NVars;
int NClasses;
int N_params;

int NHidden;
float* theta;
float* best_theta;
float* momentum_speed;
float* gradient;
float** input_to_hid;   //NHidden x NVars
float** hid_to_class;  //NHidden x NClasses

float wd_coefficient;
int NIter;
float LearnRate;
float MomentumMult;
bool do_early_stopping;
int mini_batch_size;
float best_loss;
float best_after_n_iters;

int train_batch_start;

float** hid_input;
float** hid_output;
float** class_input;
float* maxs_small;
float** class_normalizer;
float** log_class_prob;
float** class_prob;
float classification_loss;
float wd_loss;
float total_loss;

float** prob_targets;
float** hid_prob_targets;
float** inputs_hid_prob_targets;

float** gradient_input_to_hid;
float** gradient_hid_to_class;

void read_data()
{
    ifstream fin_tr("train_inputs.txt");
    ifstream fin_va("validation_inputs.txt");
    ifstream fin_te("test_inputs.txt");
    ifstream fin_tr_t("train_targets.txt");
    ifstream fin_va_t("validation_targets.txt");
    ifstream fin_te_t("test_targets.txt");

    cout<<"Reading train inputs"<<endl;
    train_inputs = new float*[NVars];
    for(int i=0;i!=NVars;i++)
        train_inputs[i] = new float[N_train];
    for(int i=0;i!=NVars;i++)
        for(int j=0;j!=N_train;j++)
            fin_tr>>train_inputs[i][j];
    cout<<"Reading validation inputs"<<endl;
    validation_inputs = new float*[NVars];
    for(int i=0;i!=NVars;i++)
        validation_inputs[i] = new float[N_validation];
    for(int i=0;i!=NVars;i++)
        for(int j=0;j!=N_validation;j++)
            fin_va>>validation_inputs[i][j];
    cout<<"Reading test inputs"<<endl;
    test_inputs = new float*[NVars];
    for(int i=0;i!=NVars;i++)
        test_inputs[i] = new float[N_test];
    for(int i=0;i!=NVars;i++)
        for(int j=0;j!=N_test;j++)
            fin_te>>test_inputs[i][j];
    cout<<"Creating batch train inputs"<<endl;
    train_batch = new float*[NVars];
    for(int i=0;i!=NVars;i++)
        train_batch[i] = new float[mini_batch_size];
    for(int i=0;i!=NVars;i++)
        for(int j=0;j!=mini_batch_size;j++)
            train_batch[i][j] = train_inputs[i][j];

    cout<<"Reading train targets"<<endl;
    train_targets = new int*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        train_targets[i] = new int[N_train];
        for(int j=0;j!=N_train;j++)
            fin_tr_t>>train_targets[i][j];
    }
    cout<<"Reading validation targets"<<endl;
    validation_targets = new int*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        validation_targets[i] = new int[N_validation];
        for(int j=0;j!=N_validation;j++)
            fin_va_t>>validation_targets[i][j];
    }
    cout<<"Reading test targets"<<endl;
    test_targets = new int*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        test_targets[i] = new int[N_test];
        for(int j=0;j!=N_test;j++)
            fin_te_t>>test_targets[i][j];
    }
    cout<<"Creating batch train targets"<<endl;
    train_batch_targets = new int*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        train_batch_targets[i] = new int[mini_batch_size];
        for(int j=0;j!=mini_batch_size;j++)
            train_batch_targets[i][j] = train_targets[i][j];
    }

}
void Clean()
{
    for(int i=0;i!=NVars;i++)
    {
        delete train_inputs[i];
        delete validation_inputs[i];
        delete test_inputs[i];
        delete train_batch[i];
    }
    delete train_inputs;
    delete validation_inputs;
    delete test_inputs;
    delete train_batch;

    for(int i=0;i!=NClasses;i++)
    {
        delete train_targets[i];
        delete validation_targets[i];
        delete test_targets[i];
        delete train_batch_targets[i];
    }
    delete train_targets;
    delete validation_targets;
    delete test_targets;
    delete train_batch_targets;

    delete theta;

    for(int i=0;i!=NHidden;i++)
    {
        delete input_to_hid[i];
    }
    delete input_to_hid;
    for(int i=0;i!=NClasses;i++)
    {
        delete hid_to_class[i];
    }
    delete hid_to_class;
    delete momentum_speed;
    delete gradient;
    delete best_theta;

    for(int i=0;i!=NHidden;i++)
    {
        delete hid_input[i];
        delete hid_output[i];
        delete hid_prob_targets[i];
        delete gradient_input_to_hid[i];
    }
    delete hid_input;
    delete hid_output;
    delete hid_prob_targets;
    delete gradient_input_to_hid;

    for(int i=0;i!=NClasses;i++)
    {
        delete class_input[i];
        delete class_normalizer[i];
        delete log_class_prob[i];
        delete class_prob[i];
        delete prob_targets[i];
        delete gradient_hid_to_class[i];
    }
    delete class_input;
    delete class_normalizer;
    delete log_class_prob;
    delete class_prob;
    delete prob_targets;
    delete gradient_hid_to_class;
    delete maxs_small;
    for(int i=0;i!=NVars;i++)
    {
        delete inputs_hid_prob_targets[i];
    }
    delete inputs_hid_prob_targets;
}
void matmul(float** a, float** b, float** c, int size1, int size2, int size3, int transpose)
{
    if(transpose == 0)
    {
        for(int i=0;i!=size1;i++)
        {
            for(int j=0;j!=size3;j++)
            {
                c[i][j] = 0;
                for(int k=0;k!=size2;k++)
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    if(transpose == 1)
    {
        for(int i=0;i!=size1;i++)
        {
            for(int j=0;j!=size3;j++)
            {
                c[i][j] = 0;
                for(int k=0;k!=size2;k++)
                {
                    c[i][j] += a[k][i] * b[k][j];
                }
            }
        }
    }
    if(transpose == 2)
    {
        for(int i=0;i!=size1;i++)
        {
            for(int j=0;j!=size3;j++)
            {
                c[i][j] = 0;
                for(int k=0;k!=size2;k++)
                {
                    c[i][j] += a[i][k] * b[j][k];
                }
            }
        }
    }
    if(transpose == 3)
    {
        for(int i=0;i!=size1;i++)
        {
            for(int j=0;j!=size3;j++)
            {
                c[i][j] = 0;
                for(int k=0;k!=size2;k++)
                {
                    c[i][j] += a[k][i] * b[j][k];
                }
            }
        }
    }
}
void theta_to_model(float* theta1, float** model_input_to_hid, float** model_hid_to_output)
{
    int counter = 0;
    for(int i=0;i!=NHidden;i++)
    {
        for(int j=0;j!=NVars;j++)
        {
            model_input_to_hid[i][j] = theta1[counter];
            counter++;
        }
    }
    for(int i=0;i!=NClasses;i++)
    {
        for(int j=0;j!=NHidden;j++)
        {
            model_hid_to_output[i][j] = theta1[counter];
            counter++;
        }
    }
}
void model_to_theta(float* theta1, float** model_input_to_hid, float** model_hid_to_output)
{
    int counter = 0;
    for(int i=0;i!=NHidden;i++)
    {
        for(int j=0;j!=NVars;j++)
        {
            theta1[counter] = model_input_to_hid[i][j];
            counter++;
        }
    }
    for(int i=0;i!=NClasses;i++)
    {
        for(int j=0;j!=NHidden;j++)
        {
            theta1[counter] = model_hid_to_output[i][j];
            counter++;
        }
    }
}
void initial_model_theta()
{
    N_params = (NVars + NClasses)*NHidden;
    theta = new float[N_params];
    momentum_speed = new float[N_params];
    gradient = new float[N_params];
    best_theta = new float[N_params];
    for(int i=0;i!=N_params;i++)
    {
        theta[i] = cos(i);
        theta[i] = theta[i] * 0.1;
        momentum_speed[i] = 0;
        gradient[i] = 0;
        best_theta[i] = 0;
    }
    int counter = 0;
    input_to_hid = new float*[NHidden];
    for(int i=0;i!=NHidden;i++)
    {
        input_to_hid[i] = new float[NVars];
        for(int j=0;j!=NVars;j++)
        {
            input_to_hid[i][j] = theta[counter];
            counter++;
        }
    }
    hid_to_class = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        hid_to_class[i] = new float[NHidden];
        for(int j=0;j!=NHidden;j++)
        {
            hid_to_class[i][j] = theta[counter];
            counter++;
        }
    }

    hid_input = new float*[NHidden];
    for(int i=0;i!=NHidden;i++)
    {
        hid_input[i] = new float[MaxSampleSize];
    }
    hid_output = new float*[NHidden];
    for(int i=0;i!=NHidden;i++)
    {
        hid_output[i] = new float[MaxSampleSize];
    }
    class_input = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        class_input[i] = new float[MaxSampleSize];
    }
    maxs_small = new float[MaxSampleSize];
    class_normalizer = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        class_normalizer[i] = new float[MaxSampleSize];
    }
    log_class_prob = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        log_class_prob[i] = new float[MaxSampleSize];
    }
    class_prob = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        class_prob[i] = new float[MaxSampleSize];
    }
    prob_targets = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        prob_targets[i] = new float[mini_batch_size];
    }
    gradient_hid_to_class = new float*[NClasses];
    for(int i=0;i!=NClasses;i++)
    {
        gradient_hid_to_class[i] = new float[NHidden];
        for(int j=0;j!=NHidden;j++)
        {
            gradient_hid_to_class[i][j] = 0;
        }
    }
    hid_prob_targets = new float*[NHidden];
    for(int i=0;i!=NHidden;i++)
    {
        hid_prob_targets[i] = new float[mini_batch_size];
    }
    inputs_hid_prob_targets = new float*[NVars];
    for(int i=0;i!=NVars;i++)
    {
        inputs_hid_prob_targets[i] = new float[NHidden];
    }
    gradient_input_to_hid = new float*[NHidden];
    for(int i=0;i!=NHidden;i++)
    {
        gradient_input_to_hid[i] = new float[NVars];
    }
}
void make_batch(int train_batch_start)
{
    for(int i=0;i!=NVars;i++)
        for(int j=0;j!=mini_batch_size;j++)
            train_batch[i][j] = train_inputs[i][train_batch_start+j];
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=mini_batch_size;j++)
            train_batch_targets[i][j] = train_targets[i][train_batch_start+j];
}
float logistic(float input)
{
    return 1./(1.+exp(-input));
}
void log_sum_exp_over_rows(int SampleSize)
{
    for(int i=0;i!=SampleSize;i++)
    {
        maxs_small[i] = class_input[0][i];
        for(int j=0;j!=NClasses;j++)
        {
            if(class_input[j][i] > maxs_small[i])
                maxs_small[i] = class_input[j][i];
        }
        for(int j=0;j!=NClasses;j++)
        {
            class_normalizer[j][i] = 0;
            for(int k=0;k!=NClasses;k++)
            {
                class_normalizer[j][i] = class_normalizer[j][i] + exp(class_input[k][i] - maxs_small[i]);
            }
            class_normalizer[j][i] = log(class_normalizer[j][i]) + maxs_small[i];
        }
    }
}
void d_loss_by_d_model()
{
    matmul(input_to_hid,train_batch,hid_input,NHidden,NVars,mini_batch_size,0);
    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=mini_batch_size;j++)
            hid_output[i][j] = logistic(hid_input[i][j]);
    matmul(hid_to_class,hid_output,class_input,NClasses,NHidden,mini_batch_size,0);
    log_sum_exp_over_rows(mini_batch_size);
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=mini_batch_size;j++)
        {
            log_class_prob[i][j] = class_input[i][j] - class_normalizer[i][j];
            class_prob[i][j] = exp(log_class_prob[i][j]);
        }

    classification_loss = 0;
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=mini_batch_size;j++)
            classification_loss += log_class_prob[i][j] * train_batch_targets[i][j];
    classification_loss = -classification_loss/mini_batch_size;

    model_to_theta(theta,input_to_hid,hid_to_class);
    wd_loss = 0;
    for(int i=0;i!=N_params;i++)
        wd_loss += theta[i]*theta[i];
    wd_loss = wd_loss/2.*wd_coefficient;
    total_loss = classification_loss + wd_loss;

    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=mini_batch_size;j++)
            prob_targets[i][j] = (class_prob[i][j] - train_batch_targets[i][j])/mini_batch_size;

    matmul(prob_targets,hid_output,gradient_hid_to_class,NClasses,mini_batch_size,NHidden,2);
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=NHidden;j++)
            gradient_hid_to_class[i][j] += wd_coefficient*hid_to_class[i][j];

    matmul(hid_to_class,prob_targets,hid_prob_targets,NHidden,NClasses,mini_batch_size,1);
    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=mini_batch_size;j++)
            hid_prob_targets[i][j] = hid_prob_targets[i][j]*(1-hid_output[i][j])*hid_output[i][j];

    matmul(train_batch,hid_prob_targets,inputs_hid_prob_targets,NVars,mini_batch_size,NHidden,2);

    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=NVars;j++)
            gradient_input_to_hid[i][j] = inputs_hid_prob_targets[j][i];

    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=NVars;j++)
            gradient_input_to_hid[i][j] = gradient_input_to_hid[i][j] + wd_coefficient*input_to_hid[i][j];
}
float loss(float** sample, int** targets, int SampleSize)
{
    matmul(input_to_hid,sample,hid_input,NHidden,NVars,SampleSize,0);
    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=SampleSize;j++)
            hid_output[i][j] = logistic(hid_input[i][j]);
    matmul(hid_to_class,hid_output,class_input,NClasses,NHidden,SampleSize,0);
    log_sum_exp_over_rows(SampleSize);
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=SampleSize;j++)
        {
            log_class_prob[i][j] = class_input[i][j] - class_normalizer[i][j];
            class_prob[i][j] = exp(log_class_prob[i][j]);
        }

    classification_loss = 0;
    for(int i=0;i!=NClasses;i++)
        for(int j=0;j!=SampleSize;j++)
            classification_loss += log_class_prob[i][j] * targets[i][j];
    classification_loss = -classification_loss/SampleSize;

    model_to_theta(theta,input_to_hid,hid_to_class);
    wd_loss = 0;
    for(int i=0;i!=N_params;i++)
        wd_loss += theta[i]*theta[i];
    wd_loss = wd_loss/2.*wd_coefficient;
    total_loss = classification_loss + wd_loss;
    return total_loss;
}
float classification_error(float** sample, int** targets, int SampleSize)
{
    float error=0;
    matmul(input_to_hid,sample,hid_input,NHidden,NVars,SampleSize,0);
    for(int i=0;i!=NHidden;i++)
        for(int j=0;j!=SampleSize;j++)
            hid_output[i][j] = logistic(hid_input[i][j]);
    matmul(hid_to_class,hid_output,class_input,NClasses,NHidden,SampleSize,0);

    for(int i=0;i!=SampleSize;i++)
    {
        float max1;
        int max1pos = 0;
        float max2;
        int max2pos = 0;
        for(int j=0;j!=NClasses;j++)
        {
            if(j == 0 || (class_input[j][i] > max1))
            {
                max1 = class_input[j][i];
                max1pos = j;
            }
            if(j == 0 || (targets[j][i] > max2))
            {
                max2 = targets[j][i];
                max2pos = j;
            }
        }
        if(max1pos != max2pos)
            error++;
    }
    return error/SampleSize;
}

int main(int argc, char* argv[])
{
    srand((unsigned)time(NULL));

    float trainloss;
    float validloss;

    N_train = 1000;
    N_validation = 1000;
    N_test = 9000;
    MaxSampleSize = max(N_train,N_validation);
    MaxSampleSize = max(N_test,MaxSampleSize);
    NClasses = 10;
    NVars = 256;
    NHidden = 36;
    mini_batch_size = 100;
    NIter = 1000;
    wd_coefficient = 0.003;
    LearnRate = 0.275;
    MomentumMult = 0.80;
    do_early_stopping = true;

    read_data();
    //print_data();

    initial_model_theta();
    theta_to_model(theta,input_to_hid,hid_to_class);
    model_to_theta(theta,input_to_hid,hid_to_class);

    ofstream fout("losses.txt");

    for(int i=0;i!=NIter;i++)
    {
        theta_to_model(theta,input_to_hid,hid_to_class);
        train_batch_start = (i*mini_batch_size) % N_train;
        make_batch(train_batch_start);
        d_loss_by_d_model();
        model_to_theta(gradient,gradient_input_to_hid,gradient_hid_to_class);
        for(int i=0;i!=N_params;i++)
            momentum_speed[i] = momentum_speed[i] * MomentumMult - gradient[i];
        for(int i=0;i!=N_params;i++)
            theta[i] = theta[i] + momentum_speed[i]*LearnRate;
        theta_to_model(theta,input_to_hid,hid_to_class);
        trainloss = loss(train_inputs,train_targets,N_train);
        validloss = loss(validation_inputs,validation_targets,N_validation);

        if(i==0 || (do_early_stopping && (validloss < best_loss)))
        {
            for(int i=0;i!=N_params;i++)
                best_theta[i] = theta[i];
            best_loss = validloss;
            best_after_n_iters = i;
        }

        cout<<i<<"\t"<<trainloss<<"\t"<<validloss<<endl;
        fout<<i<<"\t"<<trainloss<<"\t"<<validloss<<endl;
    }

    if(do_early_stopping)
    {
        for(int i=0;i!=N_params;i++)
            theta[i] = best_theta[i];
        theta_to_model(theta,input_to_hid,hid_to_class);
        trainloss = loss(train_inputs,train_targets,N_train);
        validloss = loss(validation_inputs,validation_targets,N_validation);
        cout<<"BUT! We've made early stopping, so..."<<endl;
        cout<<best_after_n_iters<<"\t"<<trainloss<<"\t"<<validloss<<endl;
        fout<<best_after_n_iters<<"\t"<<trainloss<<"\t"<<validloss<<endl;
    }

    float testloss = loss(test_inputs,test_targets,N_test);
    cout<<NIter<<"\t"<<testloss<<"\t"<<testloss<<endl;
    fout<<NIter<<"\t"<<testloss<<"\t"<<testloss<<endl;

    float error = classification_error(test_inputs,test_targets,N_test);
    cout<<"Error = "<<error;

    Clean();
    return 0;
}
