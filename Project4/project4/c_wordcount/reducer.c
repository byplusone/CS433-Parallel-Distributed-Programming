#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE 1048576

int main(int argc, char ** argv){

    char line[BUF_SIZE];
    char * word = NULL;
    int count = 0;

    while(fgets(line, BUF_SIZE - 1, stdin) != NULL){
        int i;
        for(i = strlen(line); i >= 0; i--){
            if(line[i] == '\n' || line[i] == '\r'){
                line[i] = '\0';
            }
        }

        char * token = strchr(line, '\t');
        if(token != NULL){
            int len_word = token - line;
            if(word == NULL){
                word = (char *)malloc(sizeof(char) * (len_word + 1));
                strncpy(word, line, len_word);
                word[len_word] = 0;
            }
            if(strncmp(word, line, len_word)){
                printf("%s\t%d\n", word, count);
                free(word);
                word = (char *)malloc(sizeof(char) * (len_word + 1));
                strncpy(word, line, len_word);
                word[len_word] = 0;
                count = 0;
            }
            count += atoi(token + 1);
        }
    }

    printf("%s\t%d\n", word, count);

    if(word != NULL){
        free(word);
    }

    return 0;
}

