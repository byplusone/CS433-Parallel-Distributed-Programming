#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUF_SIZE 1048576
#define DELIMITERS " \t"

int main(int argc, char ** argv){

    char line[BUF_SIZE];
    char *filepath;

    while(fgets(line, BUF_SIZE - 1, stdin) != NULL){
        int i;
        filepath = getenv("mapreduce_map_input_file");
        for(i = strlen(line); i >= 0; i--){
            if(line[i] == '\n' || line[i] == '\r'){
                line[i] = '\0';
            }
        } 

        char * token = strtok(line, DELIMITERS);
        while(token != NULL){
            printf("%s\t%s\t1\n", token, filepath);
            token = strtok(NULL, DELIMITERS);
        }
    }

    return 0;
}

