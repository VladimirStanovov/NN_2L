#include "random_numbers.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

class sample
{
public:
  //����������� ��� ����� �������������
  sample();
  ~sample();
  void Init(int NewSize, int NewNVars, int NewNClasses, int NewNFolds,
         float NewSplitRate);
  void CleanSamp();
  //������� �������� � �������
  void SetValue(int Num, int Var, float value);
  void SetNormValue(int Num, int Var, float value);
  void SetOut(int Num, int Out, float value);
  void SetClass(int Num, int Class);
  //������� ��������� ����������� ��������
  void SetMissingInput(int Num, int Var);
  void SetMissingOutput(int Num, int Out);
  //��������� ���������� � �����
  void ReadFileClassification(char* filename);
  void ReadFileRegression(char* filename);
  //����� �� ����� ���� �������
  void ShowSampleClassification();
  void ShowNormSampleClassification();
  void ShowSampleRegression();
  //����� �������� ���������� �� �������
  float GetValue(int Num,int Var);
  //����� �������� ���������� �� �������
  float GetNormValue(int Num,int Var);
  //����� �������� ������ �� �������
  float GetOutput(int Num,int Var);
  //�������� ����� ������ ��� ���������
  int GetClass(int Num);
  //��������� �������, �����-���������
  void SplitCVRandom();
  void SplitCVPredefined();
  void SplitCVStratified();
  //������� ����� �������� �� �������
  void ClassPatternsCalc();
  //������� ��������� �������
  void SplitRandom();
  void SplitStratified();
  //���������� ����� ��������� ������� ��� �����-���������
  int GetCVLearnSize(int FoldOnTest);
  //���������� ����� �������� ������� ��� �����-���������
  int GetCVTestSize(int FoldOnTest);
  //���������� ����� ��������� �������
  int GetLearnSize();
  //���������� ����� �������� �������
  int GetTestSize();
  //������� ����� �����-�������������� ���������
  int GetCVFoldNum(int Num);
  //������� ����� ����������
  int GetNVars();
  //������� ����� �������
  int GetNClasses();
  //������� ������ �������
  int GetSize();

  int GetClassPerFold(int ClassNum,int FoldNum);

  int GetClassPositions(int ClassNum,int Num);

  int GetNClassInst(int ClassNum);
  //������ ��������� �������, �����-���������
  void SetCVLearn(sample &S_CVLearn, int FoldOnTest);
  //������ �������� �������, �����-���������
  void SetCVTest(sample &S_CVTest, int FoldOnTest);
  //������ ��������� �������
  void SetLearn(sample& S_Learn);
  //������ �������� �������
  void SetTest(sample& S_Test);
  //������������ ������� �� [0,1]
  void NormalizeCV_01(int FoldOnTest);

  //����� ���������
  int Size;         //����� �������
  int NCols;        //����� ����� �������� � �������
  int NVars;        //����� �������� ������� ����������
  int NOuts;        //����� �������� �������� ����������
  int ProblemType;  //��� ������
  int NFolds;       //����� ������ ��� �����-���������
  float SplitRate; //���� ��������� �������, ��������
                    //0.7 => ��������� 70/30
  int LearnSize;    //������ ��������� �������
  int TestSize;     //������ �������� �������

  float** Inputs;  //����� ������
  float** NormInputs; //��������������� ����� ������
  float** Outputs; //������ ������
  bool** MissingInputs;    //������ ����������� ������� ��������
  bool** MissingOutputs;   //������ ����������� �������� ��������
  int* FoldSize;    //������� ������, �� ������� �����������
                    //������� ��� �����-���������
  int* CVFoldNum;   //����� �����, � ������� ��������� ���������

  //��������� ��� ����� �������������
  int NClasses;     //����� ������� � ������
  int* Classes;     //������ ������� �������
  int* NClassInst;  //���������� �������� � �������
  int** ClassPositions;  //������ ��������, ������������� ������ �������
  int** ClassPerFold;    //����� �������� ������� ��� ������ �����
  float** Range;   //��������� ���������� ��� ������������

};
