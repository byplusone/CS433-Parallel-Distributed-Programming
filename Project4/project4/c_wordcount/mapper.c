#include <stdio.h>
#include <string.h>

#define BUF_SIZE 1048576
#define DELIMITERS " \t"

int main(int argc, char ** argv){

    char line[BUF_SIZE];

    while(fgets(line, BUF_SIZE - 1, stdin) != NULL){
        int i;
        for(i = strlen(line); i >= 0; i--){
            if(line[i] == '\n' || line[i] == '\r'){
                line[i] = '\0';
            }
        }

        char * token = strtok(line, DELIMITERS);
        while(token != NULL){
            printf("%s\t1\n", token);
            token = strtok(NULL, DELIMITERS);
        }
    }

    return 0;
}

