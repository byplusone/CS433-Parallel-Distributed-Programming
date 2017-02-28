#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE 1048576

int main(int argc, char ** argv){

    char line[BUF_SIZE];
    char line2[BUF_SIZE];
    char * word = NULL;
    char * addr = NULL;
    int count = 0;

    while(fgets(line, BUF_SIZE - 1, stdin) != NULL){
        int i;
        for(i = strlen(line); i >= 0; i--){
            if(line[i] == '\n' || line[i] == '\r'){
                line[i] = '\0';
            }
        }

        char * token = strchr(line, '\t');
        int len_word = token - line;
        strncpy(line2, token+1, strlen(line)-len_word);
        char * token2 = strchr(line2, '\t');
        int len_addr = token2 - line2;

        if(token != NULL){
            if(word == NULL){
                word = (char *)malloc(sizeof(char) * (len_word + 1));
                strncpy(word, line, len_word);
		        word[len_word] = 0;
                printf("%s", word);
            }
            if(addr == NULL){
                addr = (char *)malloc(sizeof(char) * (len_addr + 1));
                strncpy(addr, line2, len_addr);
                addr[len_addr] = 0;
            }
            if(strncmp(word, line, len_word) != 0){
                printf("\t%s:%d\n", addr, count);

                free(word);
                word = (char *)malloc(sizeof(char) * (len_word + 1));
                strncpy(word, line, len_word);
                word[len_word] = 0;
                printf("%s", word);

                free(addr);
                addr = (char *)malloc(sizeof(char) * (len_addr + 1));
                strncpy(addr, line2, len_addr);
                addr[len_addr] = 0;
                count = 0;
            }
            else if(strncmp(addr, line2, len_addr) != 0){
                printf("\t%s:%d", addr, count);
                free(addr);
                addr = (char *)malloc(sizeof(char) * (len_addr + 1));
                strncpy(addr, line2, len_addr);
                addr[len_addr] = 0;
                count = 0;
            }
            count++;
        }
    }

    printf("\t%s:%d\n", addr, count);

    if(word != NULL){
        free(word);
    }

    if(addr != NULL){
        free(addr);
    }

    return 0;
}

