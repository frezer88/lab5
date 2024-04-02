#include <stdlib.h>
#include <mpi.h>
#include <cstring>
#include <iostream>
#include <ctime>
#include <iomanip>

using namespace std;

const int MATRIX_SIZE = 3; // Размерность системы уравнений

// Функция для печати результатов
void PrintResults(const char* description, float *solutions){
    cout << description << endl;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        cout << "x[" << i << "] = " << solutions[i] << endl;
    }
}

// Функция для вычисления результатов системы уравнений
void ComputeSolutions(float *gaussMatrix, float *solutions) {
    for (int i = MATRIX_SIZE - 1; i >= 0; i--) {
        solutions[i] = gaussMatrix[i * (MATRIX_SIZE + 1) + MATRIX_SIZE];

        for (int j = i + 1; j < MATRIX_SIZE; j++) {
            solutions[i] -= gaussMatrix[i * (MATRIX_SIZE + 1) + j] * solutions[j];
        }

        solutions[i] /= gaussMatrix[i * (MATRIX_SIZE + 1) + i];
    }
}

void GaussianMethod(int processRank, int rowsPerProcess, float *pivotRow, float *partialMatrix) { // Прямой ход метода Гаусса и распределение строк

    int pivotElement, scaleElement, targetColumn, localStartRow;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        int pivotOwner = i / rowsPerProcess;
        if (processRank == pivotOwner) {
            // Выбираем строку-опору для нормализации
            int localRow = i % rowsPerProcess;
            pivotElement = partialMatrix[localRow * (MATRIX_SIZE + 1) + i];

            // Нормализация строки-опоры
            for (int j = i; j < MATRIX_SIZE + 1; ++j) {
                partialMatrix[localRow * (MATRIX_SIZE + 1) + j] /= pivotElement;
            }

            // Копируем нормализованную строку в массив pivotRow
            memcpy(pivotRow, &partialMatrix[localRow * (MATRIX_SIZE + 1)], (MATRIX_SIZE + 1) * sizeof(float));
        }

        // Рассылка нормализованной строки-опоры всем процессам
        MPI_Bcast(pivotRow, MATRIX_SIZE + 1, MPI_FLOAT, pivotOwner, MPI_COMM_WORLD);

        // Вычитание строки-опоры из всех оставшихся строк в подматрице каждого процесса
        if (processRank != pivotOwner) {
            for (int j = 0; j < rowsPerProcess; ++j) {
                int globalRowIdx = processRank * rowsPerProcess + j;
                if (globalRowIdx > i) {
                    scaleElement = partialMatrix[j * (MATRIX_SIZE + 1) + i];
                    for (int k = i; k < MATRIX_SIZE + 1; ++k) {
                        partialMatrix[j * (MATRIX_SIZE + 1) + k] -= scaleElement * pivotRow[k];
                    }
                }
            }
        }
    }
}

void ClearMemory(int processRank, float *partialMatrix, float *pivotRow, float *fullMatrix, float *computedResults){
    if (processRank == 0) {
        delete[] fullMatrix;
        delete[] computedResults;
    }
    delete[] partialMatrix;
    delete[] pivotRow;
}

int main(int argc, char *argv[]) {
    double startTime, endTime, totalTime; // Переменные для отслеживания времени выполнения
    int processRank, numProcesses; // Переменные для хранения ранга и общего количества процессов
    double globalStartTime, globalEndTime, globalTotalTime; // Переменные для глобального времени выполнения

    MPI_Init(&argc, &argv); // Инициализация MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank); // Получение ранга текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses); // Получение общего числа процессов

    if (processRank == 0) {
            globalStartTime = MPI_Wtime();
    }

    int rowsPerProcess = MATRIX_SIZE / numProcesses; // Вычисление числа строк на процесс
    float *fullMatrix, *computedResults; // Объявление матрицы системы уравнений и массива для решений

    // Инициализация матрицы в процессе с рангом 0
    if (processRank == 0) {
        fullMatrix = new float[MATRIX_SIZE * (MATRIX_SIZE + 1)]; // Выделение памяти для полной матрицы
        float predefinedMatrix[MATRIX_SIZE * (MATRIX_SIZE + 1)] = {1, 2, 3, 3, 3, 5, 7, 0, 1, 3, 4, 1}; // Заданная матрица
        std::copy(predefinedMatrix, predefinedMatrix + MATRIX_SIZE * (MATRIX_SIZE + 1), fullMatrix); // Копирование заданной матрицы
        computedResults = new float[MATRIX_SIZE]; // Выделение памяти для результата
    } else {
        fullMatrix = new float[MATRIX_SIZE * (MATRIX_SIZE + 1)]; // Выделение памяти для полной матрицы в других процессах
        computedResults = nullptr; // В других процессах массив результатов не используется
    }

    float *partialMatrix = new float[(MATRIX_SIZE + 1) * rowsPerProcess]; // Выделение памяти для подматрицы
    MPI_Scatter(fullMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, partialMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD); // Распределение матрицы по процессам

    float *pivotRow = new float[MATRIX_SIZE + 1]; // Массив для строки-опоры

    GaussianMethod(processRank, rowsPerProcess, pivotRow, partialMatrix); // Выполнение метода Гаусса

    MPI_Gather(partialMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, fullMatrix, (MATRIX_SIZE + 1) * rowsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD); // Сбор подматриц в полную матрицу

    if (processRank == 0) {
        globalEndTime = MPI_Wtime();
        globalTotalTime = globalEndTime - globalStartTime;
        cout << "Время выполнения: " << globalTotalTime << " seconds" << endl;
    }    
  

    // Обработка и вывод результатов в процессе с рангом 0
    if (processRank == 0) {
        ComputeSolutions(fullMatrix, computedResults); // Вычисление решений
        PrintResults("Результат выполнения:", computedResults); // Вывод решений
    }

    // Освобождение памяти
    ClearMemory(processRank, partialMatrix, pivotRow, fullMatrix, computedResults);

    MPI_Finalize(); // Завершение работы с MPI

    return 0;
}
