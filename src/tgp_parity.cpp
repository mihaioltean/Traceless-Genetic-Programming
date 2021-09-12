//---------------------------------------------------------------------------
//  Traceless Genetic Programming - basic source code for solving even-parity problems
//  (c) Mihai Oltean mihai.oltean@gmail.com
//  github.com/mihaioltean/genetic-programming
//  Last update on: 2016.07.01.0
//  MIT License

//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//---------------------------------------------------------------------------

//  papers to read:
//  Oltean, M., Solving Even-Parity Problems using Traceless Genetic Programming, IEEE Congress on Evolutionary Computation, Portland, 19-23 June, edited by G. Greenwood (et. al), pages 1813-1819, IEEE Press, 2004.

//  More info at:
//     github.com/mihaioltean/genetic-programming

//  Compiled with Microsoft Visual C++ 2013
//  Also compiled with XCode 7.

//  Please reports any sugestions and/or bugs to mihai.oltean@gmail.com

//  Training data file must have the following format:

//  x11 x12 ... x1n f1
//  x21 x22 ....x2n f2
//  .............
//  xm1 xm2 ... xmn fm

//  where m is the number of training data
//  and n is the number of variables.
//  xij are the inputs
//  fi are the outputs


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define NumberOfOperators 4

// AND -1
// OR -2
// NAND -3
// NOR -4
//---------------------------------------------------------------------------
struct t_tgp_chromosome{
    char *value;  // value of the current program for kth data (training, validation or test)
    
    int fitness;           //num incorrect classified
} ;
//---------------------------------------------------------------------------
struct t_tgp_parameters{
    int num_generations;
    int pop_size;                // population size
    double insertion_probability, crossover_probability;
};
//---------------------------------------------------------------------------
void allocate_training_data(char **&data, char *&target, int num_training_data, int num_variables)
{
    target = new char[num_training_data];
    data = new char*[num_training_data];
    for (int i = 0; i < num_training_data; i++)
        data[i] = new char[num_variables];
}
//---------------------------------------------------------------------------
void delete_data(char **&data, char *&target, int num_training_data)
{
    if (data)
        for (int i = 0; i < num_training_data; i++)
            delete[] data[i];
    delete[] data;
    delete[] target;
}
//---------------------------------------------------------------------------
void alocate_population(t_tgp_chromosome *&pop, int pop_size, int num_training_data)
{
    pop = new t_tgp_chromosome[pop_size];
    
    for (int i = 0; i < pop_size; i++)
        pop[i].value = new char[num_training_data];
}
//---------------------------------------------------------------------------
void copy_chromosome(t_tgp_chromosome& dest, t_tgp_chromosome& source, int num_training_data)
{
    for (int i = 0; i < num_training_data; i++)
        dest.value[i] = source.value[i];
    dest.fitness = source.fitness;
}
//---------------------------------------------------------------------------
void fitness(t_tgp_chromosome &c, int num_training_data, char **training_data, char *target)
{
    c.fitness = 0;
    for (int i = 0; i < num_training_data; i++)
        if (abs(c.value[i] - target[i])){
            c.fitness++;
    }
}
//---------------------------------------------------------------------------
void init_chromosome(t_tgp_chromosome &c, int num_variables, int num_training_data, char ** data)
{
    int random_var = rand() % num_variables;
    
    for (int i = 0; i < num_training_data; i++)
        c.value[i] = data[i][random_var];
}
//---------------------------------------------------------------------------
int sort_function(const void *a, const void *b)
{
    if (((t_tgp_chromosome *)a)->fitness > ((t_tgp_chromosome *)b)->fitness)
        return 1;
    else
        if (((t_tgp_chromosome *)a)->fitness < ((t_tgp_chromosome *)b)->fitness)
            return -1;
        else
            return 0;//!!!
}
//---------------------------------------------------------------------------
void sort_by_fitness(t_tgp_chromosome *pop, int pop_size)
// sort ascendingly the individuals in population
{
    qsort((void *)pop, pop_size, sizeof(pop[0]), sort_function);
}
//---------------------------------------------------------------------------
void free_pop_memory(t_tgp_chromosome *&pop, int pop_size)
{
    for (int i = 0; i < pop_size; i++)
        delete[] pop[i].value;
    
    delete[] pop;
}
//---------------------------------------------------------------------------
int tournament_selection(t_tgp_chromosome *pop, int pop_size, int tournament_size)
{
    int r, p;
    p = rand() % pop_size;
    for (int i = 1; i < tournament_size; i++) {
        r = rand() % pop_size;
        p = pop[r].fitness < pop[p].fitness ? r : p;
    }
    return p;
}
//---------------------------------------------------------------------------
void start_steady_state_tgp(t_tgp_parameters &parameters, char **training_data, char *target, int num_training_data, int num_variables)
{
    t_tgp_chromosome* current_pop, *new_pop;
    
    alocate_population(current_pop, parameters.pop_size, num_training_data);
    alocate_population(new_pop, parameters.pop_size, num_training_data);
    for (int i = 0; i < parameters.pop_size; i++){
        init_chromosome(current_pop[i], num_variables, num_training_data, training_data);
        fitness(current_pop[i], num_training_data, training_data, target);
    }
    
    sort_by_fitness(current_pop, parameters.pop_size);
    
    for (int g = 1; g < parameters.num_generations; g++){
        // elitism: copy best to the new population
        copy_chromosome(new_pop[0], current_pop[0], num_training_data);
        int new_pop_size = 1;
        
        if (g % 100 == 0)
            printf("%d %d\n", g, current_pop[0].fitness);
        while (new_pop_size < parameters.pop_size){
            double p = rand() / (double)RAND_MAX;
            
            if (p < parameters.insertion_probability){  // insertion of a simple program (made from a single variable)
                init_chromosome(new_pop[new_pop_size], num_variables, num_training_data, training_data);
                fitness(new_pop[new_pop_size], num_training_data, training_data, target);
                new_pop_size++;
            }
            else{  // recombination of 2 programs
                // first I have to choose an operator
                int op = rand() % NumberOfOperators;
                int p1, p2;
                double ps;
                switch (op){
                    case 0: // and
                        p1 = tournament_selection(current_pop, parameters.pop_size, 1);
                        p2 = tournament_selection(current_pop, parameters.pop_size, 1);
                        ps = rand() / (double) RAND_MAX;
                        if (ps <= parameters.crossover_probability){
                            
                            for (int i = 0; i < num_training_data; i++)
                                new_pop[new_pop_size].value[i] = current_pop[p1].value[i] & current_pop[p2].value[i];
                            
                            
                            fitness(new_pop[new_pop_size], num_training_data, training_data, target);
                            new_pop_size++;
                        }
                        else{
                            // copy one of the parents to the new population
                            copy_chromosome(new_pop[new_pop_size], current_pop[p1], num_training_data);
                            new_pop_size++;
                        }
                        break;
                        
                    case 1: // or
                        p1 = tournament_selection(current_pop, parameters.pop_size, 1);
                        p2 = tournament_selection(current_pop, parameters.pop_size, 1);
                        ps = rand() / (double) RAND_MAX;
                        if (ps <= parameters.crossover_probability){
                            for (int i = 0; i < num_training_data; i++)
                                new_pop[new_pop_size].value[i] = current_pop[p1].value[i] | current_pop[p2].value[i];
                            
                            fitness(new_pop[new_pop_size], num_training_data, training_data, target);
                            new_pop_size++;
                        }
                        else{
                            // copy one of the parents to the new population
                            copy_chromosome(new_pop[new_pop_size], current_pop[p1], num_training_data);
                            new_pop_size++;
                        }
                        break;
                    case 2: // nand
                        p1 = tournament_selection(current_pop, parameters.pop_size, 1);
                        p2 = tournament_selection(current_pop, parameters.pop_size, 1);
                        ps = rand() / (double) RAND_MAX;
                        if (ps <= parameters.crossover_probability){
                            
                            for (int i = 0; i < num_training_data; i++)
                                new_pop[new_pop_size].value[i] = !(current_pop[p1].value[i] & current_pop[p2].value[i]);
                            
                            fitness(new_pop[new_pop_size], num_training_data, training_data, target);
                            new_pop_size++;
                        }
                        else{
                            // copy one of the parents to the new population
                            copy_chromosome(new_pop[new_pop_size], current_pop[p1], num_training_data);
                            new_pop_size++;
                        }
                        break;
                    case 3: // nor
                        p1 = tournament_selection(current_pop, parameters.pop_size, 1);
                        p2 = tournament_selection(current_pop, parameters.pop_size, 1);
                        ps = rand() / (double) RAND_MAX;
                        if (ps <= parameters.crossover_probability){
                            
                            for (int i = 0; i < num_training_data; i++)
                                new_pop[new_pop_size].value[i] = !(current_pop[p1].value[i] | current_pop[p2].value[i]);
                            
                            fitness(new_pop[new_pop_size], num_training_data, training_data, target);
                            new_pop_size++;
                        }
                        else{
                            // copy one of the parents to the new population
                            copy_chromosome(new_pop[new_pop_size], current_pop[p1], num_training_data);
                            new_pop_size++;
                        }
                        break;
                        
                }// switch
            }
        }
        
        for (int k = 0; k < parameters.pop_size; k++)
            copy_chromosome(current_pop[k], new_pop[k], num_training_data);
        sort_by_fitness(current_pop, parameters.pop_size);
        
    }
}
//---------------------------------------------------------------------------
bool read_training_data(const char *filename, char **&training_data, char *&target, int &num_training_data, int &num_variables)
{
    FILE* f = fopen(filename, "r");
    if (!f)
        return false;
    
    fscanf(f, "%d%d", &num_training_data, &num_variables);
    
    allocate_training_data(training_data, target, num_training_data, num_variables);
    
    int v;
    for (int i = 0; i < num_training_data; i++) {
        for (int j = 0; j < num_variables; j++){
            
            fscanf(f, "%d", &v);
            training_data[i][j] = v;
        }
        fscanf(f, "%d", &v);
        target[i] = v;
    }
    fclose(f);
    return true;
}
//---------------------------------------------------------------------------
int main(void)
{
    
    t_tgp_parameters params;
    
    params.pop_size = 100;						    // the number of individuals in population
    params.num_generations = 10000;					// the number of generations
    params.insertion_probability = 0.1;              // insertion probability
    params.crossover_probability = 0.9;             // crossover probability
    
    
    int num_training_data, num_variables;
    char** training_data;
    char *target;
    
    if (!read_training_data("dataset//even_5_parity.txt", training_data, target, num_training_data, num_variables)) {
        printf("Cannot find input file! Please specify the correct (full) path!");
        getchar();
        return 1;
    }
    
    
    printf("num training data = %d\n", num_training_data);
    printf("num variables = %d\n", num_variables);
    
    srand(0);
    start_steady_state_tgp( params, training_data, target, num_training_data, num_variables);
    
    delete_data(training_data, target, num_training_data);
    printf("Press enter ...");
    getchar();
    
    
    return 0;
}
//---------------------------------------------------------------------------
