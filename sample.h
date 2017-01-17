#include "random_numbers.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

class sample
{
public:
  //конструктор для задач классификации
  sample();
  ~sample();
  void Init(int NewSize, int NewNVars, int NewNClasses, int NewNFolds,
         float NewSplitRate);
  void CleanSamp();
  //задание значений в выборке
  void SetValue(int Num, int Var, float value);
  void SetNormValue(int Num, int Var, float value);
  void SetOut(int Num, int Out, float value);
  void SetClass(int Num, int Class);
  //задание положений пропущенных значений
  void SetMissingInput(int Num, int Var);
  void SetMissingOutput(int Num, int Out);
  //первичное считывание с файла
  void ReadFileClassification(char* filename);
  void ReadFileRegression(char* filename);
  //вывод на экран всей выборки
  void ShowSampleClassification();
  void ShowNormSampleClassification();
  void ShowSampleRegression();
  //взять значение переменной из выборки
  float GetValue(int Num,int Var);
  //взять значение переменной из выборки
  float GetNormValue(int Num,int Var);
  //взять значение выхода из выборки
  float GetOutput(int Num,int Var);
  //получить номер класса для измерения
  int GetClass(int Num);
  //разбиение выборки, кросс-валидация
  void SplitCVRandom();
  void SplitCVPredefined();
  void SplitCVStratified();
  //считает число объектов по классам
  void ClassPatternsCalc();
  //простое разбиение выборки
  void SplitRandom();
  void SplitStratified();
  //возвращает объем обучающей выборки для кросс-валидации
  int GetCVLearnSize(int FoldOnTest);
  //возвращает объем тестовой выборки для кросс-валидации
  int GetCVTestSize(int FoldOnTest);
  //возвращает объем обучающей выборки
  int GetLearnSize();
  //возвращает объем тестовой выборки
  int GetTestSize();
  //вернуть номер кросс-валидационного разбиения
  int GetCVFoldNum(int Num);
  //вернуть число переменных
  int GetNVars();
  //вернуть число классов
  int GetNClasses();
  //вернуть размер выборки
  int GetSize();

  int GetClassPerFold(int ClassNum,int FoldNum);

  int GetClassPositions(int ClassNum,int Num);

  int GetNClassInst(int ClassNum);
  //Задать обучающую выборку, кросс-валидация
  void SetCVLearn(sample &S_CVLearn, int FoldOnTest);
  //Задать тестовую выборку, кросс-валидация
  void SetCVTest(sample &S_CVTest, int FoldOnTest);
  //Задать обучающую выборку
  void SetLearn(sample& S_Learn);
  //Задать тестовую выборку
  void SetTest(sample& S_Test);
  //нормализация выборки на [0,1]
  void NormalizeCV_01(int FoldOnTest);

  //общие параметры
  int Size;         //объем выборки
  int NCols;        //общее число столбцов в выборке
  int NVars;        //число столбцов входных параметров
  int NOuts;        //число столбцов выходных параметров
  int ProblemType;  //тип задачи
  int NFolds;       //число частей при кросс-валидации
  float SplitRate; //доля обучающей выборки, например
                    //0.7 => разбиение 70/30
  int LearnSize;    //размер обучающей выборки
  int TestSize;     //размер тестовой выборки

  float** Inputs;  //входы задачи
  float** NormInputs; //Нормализованные входы задачи
  float** Outputs; //выходы задачи
  bool** MissingInputs;    //массив пропущенных входных значений
  bool** MissingOutputs;   //массив пропущенных выходных значений
  int* FoldSize;    //размеры частей, на которые разбивается
                    //выборка при кросс-валидации
  int* CVFoldNum;   //номер части, к которой относится измерение

  //параметры для задач классификации
  int NClasses;     //число классов в задаче
  int* Classes;     //массив номеров классов
  int* NClassInst;  //количество объектов в классах
  int** ClassPositions;  //Номера объектов, принадлежащих разным классам
  int** ClassPerFold;    //число объектов классов для каждой части
  float** Range;   //диапазоны переменных для нормализации

};
