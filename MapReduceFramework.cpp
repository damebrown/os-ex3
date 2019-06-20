//
// Created by daniel.brown1 on 5/2/19.
//

#include "MapReduceFramework.h"
#include <stdlib.h>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <queue>
#include <semaphore.h>


typedef void *JobHandle;

class JobContext;

class ThreadContext {
public:
    int _id;
    JobContext *_jc;
    IntermediateVec _threadInter;
    ThreadContext(int threadID, JobContext *jobContext) {
        _id = threadID;
        _jc = jobContext;
        _threadInter = {};
    }
    ~ThreadContext() = default;
};

class JobContext {
public:
    const MapReduceClient *_client;
    const InputVec *_inputVec;
    OutputVec *_outputVec;
    int _threads_num;
    std::atomic<int> *_atomic_counter;
    //IntermediateVec *_pairs;
    pthread_mutex_t _mutex;
    bool _isShuffle;
    std::queue<IntermediateVec> _works;
    sem_t _semaphore;
    Barrier *_barrier;
    pthread_t *_threads;
    vector<ThreadContext *> _tcs;
    JobState _jobState;
    std::atomic<int> *_works_to_do_counter;
    std::atomic<int> *_done_works_counter;

    JobContext(const MapReduceClient *client, const InputVec *inputVec, OutputVec *outputVec, const int threads_num);
    ~JobContext() {
        for (auto thr : _tcs){
            delete thr;
        }
//        for (int i = 1; i < _threads_num; ++i){
//            delete &_threads[i];
//        }
        delete _threads;
        delete _barrier;
        delete _atomic_counter;
        delete _works_to_do_counter;
        delete _done_works_counter;
    };
};

JobContext::JobContext(const MapReduceClient *client, const InputVec *inputVec, OutputVec *outputVec,
                       const int threads_num) {
    _client = client;
    _inputVec = inputVec;
    _outputVec = outputVec;
    _threads_num = threads_num;
    _barrier = new Barrier(_threads_num);
    _atomic_counter = new std::atomic<int>(0);
    _works_to_do_counter = new std::atomic<int>(0);
    _done_works_counter = new std::atomic<int>(0);
    _mutex = PTHREAD_MUTEX_INITIALIZER;
    _isShuffle = true;
    _works = {};
    sem_init(&_semaphore, 0, 0);
    _threads = new pthread_t[_threads_num];
    _tcs = {};
    _jobState = {UNDEFINED_STAGE, 0};
}

//JobContext* curJob;

//pthread_t *threads_factorty();

void emit2(K2 *key, V2 *value, void *context) {
    ThreadContext *threadContext = (ThreadContext *) context;
    threadContext->_threadInter.push_back({key, value});
}


void emit3(K3 *key, V3 *value, void *context) {
    JobContext *jc = (JobContext *) context;
    jc->_outputVec->push_back({key, value});
    (*jc->_done_works_counter)++;
}

bool sort_by_first(const IntermediatePair &a, const IntermediatePair &b) {
    return (a.first->operator<(*b.first));
}

void thread_sort_phase(ThreadContext *tc) {
    sort(tc->_threadInter.begin(), tc->_threadInter.end(), sort_by_first);
}

void shuffle_phase(JobContext *jc) {
    //std::atomic<int> *_atomic_counter(0);
    jc->_jobState.stage = REDUCE_STAGE;
    jc->_jobState.percentage = 0;
    std::vector<IntermediateVec *> threads_vec;
    for (int i = 0; i < jc->_threads_num; ++i) {
        threads_vec.push_back(&jc->_tcs[i]->_threadInter);
    }
    while (!threads_vec.empty()) {
        IntermediateVec firsts = {};
        for (vector<IntermediateVec *>::iterator iter = threads_vec.begin();
             iter != threads_vec.end(); ++iter) {
            if ((*iter)->empty()) {
                threads_vec.erase(iter);
                --iter; //********************
            } else {
                firsts.push_back((*iter)->at(0));
            }
            if (iter == threads_vec.end()) {
                break;
            }
        }
        if (!firsts.empty()) {
            sort(firsts.begin(), firsts.end(), sort_by_first);
            K2 *max = firsts.at(0).first;
            //---------------------------------
            IntermediateVec work = {};
            pthread_mutex_lock(&jc->_mutex);

            for (auto inter : threads_vec) {
                if ((!((inter)->at(0).first->operator<(*max))) &&
                    (!(max->operator<(*(inter)->at(0).first)))) {
                    work.push_back((*inter).at(0));
                    (*inter).erase((*inter).begin());
                }
            }
            jc->_works.push(work);
            (*jc->_works_to_do_counter)++;
            sem_post(&(jc->_semaphore));
            firsts = {};
            pthread_mutex_unlock(&jc->_mutex);
        }
    }
    for (int i = 0; i < jc->_threads_num; ++i) {
        sem_post(&(jc->_semaphore));
    }
    jc->_isShuffle = false;
}

void reduce_phase(JobContext *jc) {
    while (!jc->_works.empty() || jc->_isShuffle) {
        sem_wait(&(jc->_semaphore));
        pthread_mutex_lock(&jc->_mutex);
        if(jc->_jobState.percentage < 100) {
            jc->_client->reduce(&jc->_works.front(), jc);
            jc->_works.pop();
        }
        pthread_mutex_unlock(&jc->_mutex);
        if (!jc->_isShuffle) {
            jc->_jobState.percentage = float(jc->_done_works_counter->load()) / float(jc->_works_to_do_counter->load()) * float(100);
        }

    }
}

void *thread_function(void *arg) {
    ThreadContext *tc = (ThreadContext *) arg;
    JobContext *jc = tc->_jc;
    int old = 0;
    int size = (int) jc->_inputVec->size();
    while (jc->_atomic_counter->load() < size) {
        jc->_jobState.stage = MAP_STAGE;
        old = (*jc->_atomic_counter)++;
        jc->_client->map(jc->_inputVec->at((unsigned long) old).first,
                         jc->_inputVec->at((unsigned long) old).second, tc);
        jc->_jobState.percentage = (old / size) * 100;
    }
    thread_sort_phase(tc);
    jc->_barrier->barrier();
    if (tc->_id == 0) {
        shuffle_phase(jc);
    }
    reduce_phase(jc);
    return tc;
}

/**
 * This function starts running the MapReduce algorithm (with several
threads) and returns a JobHandle.
 * @param client– The implementation of MapReduceClient or in other words the task that the
 * framework should run.
 * @param inputVec– a vector of type std::vector<std::pair<K1*, V1*>>, the input elements
 * @param outputVec– a vector of type std::vector<std::pair<K3*, V3*>>, to which the output
 * elements will be added before returning. You can assume that outputVec is empty.
 * @param multiThreadLevel– the number of worker threads to be used for running the algorithm. You
 * can assume this argument is valid (greater or equal to 1).
 * @return a JobHandle.
 */
JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel) {
    if (inputVec.empty()){
        return nullptr;
    }
    auto curJob = new JobContext(&client, &inputVec, &outputVec, multiThreadLevel);

    for (int i = 0; i < multiThreadLevel; ++i) {
        curJob->_tcs.push_back(new ThreadContext(i, curJob));
        pthread_create(curJob->_threads + i, nullptr, thread_function, curJob->_tcs[i]);
    }
    auto jh = (void *) curJob;
    return jh;
}

/**
 * a function gets the job handle returned by startMapReduceJob and waits until
 * it is finished.
 * @param job
 */
void waitForJob(JobHandle job) {
    if (job == NULL){
        return;
    }
    auto jc = (JobContext *) job;
    if (jc != nullptr) {
        for (int i = 0; i < jc->_threads_num; ++i) {
            pthread_join(*(jc->_threads + i), nullptr);
        }
        jc->_jobState.stage = UNDEFINED_STAGE;
    }
}

/**
 * this function gets a job handle and check for his current state in a given
 * JobState struct.
 * @param job
 * @param state
 */
void getJobState(JobHandle job, JobState *state) {
    if (job == NULL){
        return;
    }
    auto jc = (JobContext *) job;
    *state = jc->_jobState;
}

/**
 * Releasing all resources of a job. You should prevent releasing resources
 * before the job finished. After this function is called the job handle will be invalid.
 * @param job
 */
void closeJobHandle(JobHandle job) {
    if (job == NULL){
        return;
    }
    auto jc = (JobContext *) job;
    delete jc;
}