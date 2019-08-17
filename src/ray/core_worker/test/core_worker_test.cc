#include <thread>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ray/common/buffer.h"
#include "ray/core_worker/context.h"
#include "ray/core_worker/core_worker.h"
#include "ray/core_worker/transport/direct_actor_transport.h"

#include "ray/core_worker/store_provider/local_plasma_provider.h"
#include "ray/core_worker/store_provider/memory_store_provider.h"

#include "ray/rpc/raylet/raylet_client.h"
#include "src/ray/protobuf/direct_actor.grpc.pb.h"
#include "src/ray/protobuf/direct_actor.pb.h"
#include "src/ray/util/test_util.h"

#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/bind.hpp>

#include "ray/thirdparty/hiredis/async.h"
#include "ray/thirdparty/hiredis/hiredis.h"
#include "ray/util/test_util.h"

namespace ray {

std::string store_executable;
std::string raylet_executable;
std::string mock_worker_executable;

ray::ObjectID RandomObjectID() { return ObjectID::FromRandom(); }

static void flushall_redis(void) {
  redisContext *context = redisConnect("127.0.0.1", 6379);
  freeReplyObject(redisCommand(context, "FLUSHALL"));
  freeReplyObject(redisCommand(context, "SET NumRedisShards 1"));
  freeReplyObject(redisCommand(context, "LPUSH RedisShards 127.0.0.1:6380"));
  redisFree(context);
}

std::shared_ptr<Buffer> GenerateRandomBuffer() {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(1, 10);
  std::uniform_int_distribution<> value_dis(1, 255);

  std::vector<uint8_t> arg1(dis(gen), value_dis(gen));
  return std::make_shared<LocalMemoryBuffer>(arg1.data(), arg1.size(), true);
}

std::unique_ptr<ActorHandle> CreateActorHelper(
    CoreWorker &worker, const std::unordered_map<std::string, double> &resources,
    bool is_direct_call, uint64_t max_reconstructions) {
  std::unique_ptr<ActorHandle> actor_handle;

  // Test creating actor.
  uint8_t array[] = {1, 2, 3};
  auto buffer = std::make_shared<LocalMemoryBuffer>(array, sizeof(array));

  RayFunction func{ray::Language::PYTHON, {"actor creation task"}};
  std::vector<TaskArg> args;
  args.emplace_back(TaskArg::PassByValue(buffer));

  ActorCreationOptions actor_options{max_reconstructions, is_direct_call, resources, {}};

  // Create an actor.
  RAY_CHECK_OK(worker.Tasks().CreateActor(func, args, actor_options, &actor_handle));
  return actor_handle;
}

class CoreWorkerTest : public ::testing::Test {
 public:
  CoreWorkerTest(int num_nodes) : gcs_options_("127.0.0.1", 6379, "") {
    // flush redis first.
    flushall_redis();

    RAY_CHECK(num_nodes >= 0);
    if (num_nodes > 0) {
      raylet_socket_names_.resize(num_nodes);
      raylet_store_socket_names_.resize(num_nodes);
    }

    // start plasma store.
    for (auto &store_socket : raylet_store_socket_names_) {
      store_socket = StartStore();
    }

    // start raylet on each node. Assign each node with different resources so that
    // a task can be scheduled to the desired node.
    for (int i = 0; i < num_nodes; i++) {
      raylet_socket_names_[i] =
          StartRaylet(raylet_store_socket_names_[i], "127.0.0.1", "127.0.0.1",
                      "\"CPU,4.0,resource" + std::to_string(i) + ",10\"");
    }
  }

  ~CoreWorkerTest() {
    for (const auto &raylet_socket : raylet_socket_names_) {
      StopRaylet(raylet_socket);
    }

    for (const auto &store_socket : raylet_store_socket_names_) {
      StopStore(store_socket);
    }
  }

  JobID NextJobId() const {
    static uint32_t job_counter = 1;
    return JobID::FromInt(job_counter++);
  }

  std::string StartStore() {
    std::string store_socket_name = "/tmp/store" + RandomObjectID().Hex();
    std::string store_pid = store_socket_name + ".pid";
    std::string plasma_command = store_executable + " -m 10000000 -s " +
                                 store_socket_name +
                                 " 1> /dev/null 2> /dev/null & echo $! > " + store_pid;
    RAY_LOG(DEBUG) << plasma_command;
    RAY_CHECK(system(plasma_command.c_str()) == 0);
    usleep(200 * 1000);
    return store_socket_name;
  }

  void StopStore(std::string store_socket_name) {
    std::string store_pid = store_socket_name + ".pid";
    std::string kill_9 = "kill -9 `cat " + store_pid + "`";
    RAY_LOG(DEBUG) << kill_9;
    ASSERT_EQ(system(kill_9.c_str()), 0);
    ASSERT_EQ(system(("rm -rf " + store_socket_name).c_str()), 0);
    ASSERT_EQ(system(("rm -rf " + store_socket_name + ".pid").c_str()), 0);
  }

  std::string StartRaylet(std::string store_socket_name, std::string node_ip_address,
                          std::string redis_address, std::string resource) {
    std::string raylet_socket_name = "/tmp/raylet" + RandomObjectID().Hex();
    std::string ray_start_cmd = raylet_executable;
    ray_start_cmd.append(" --raylet_socket_name=" + raylet_socket_name)
        .append(" --store_socket_name=" + store_socket_name)
        .append(" --object_manager_port=0 --node_manager_port=0")
        .append(" --node_ip_address=" + node_ip_address)
        .append(" --redis_address=" + redis_address)
        .append(" --redis_port=6379")
        .append(" --num_initial_workers=1")
        .append(" --maximum_startup_concurrency=10")
        .append(" --static_resource_list=" + resource)
        .append(" --python_worker_command=\"" + mock_worker_executable + " " +
                store_socket_name + " " + raylet_socket_name + "\"")
        .append(" --config_list=initial_reconstruction_timeout_milliseconds,2000")
        .append(" & echo $! > " + raylet_socket_name + ".pid");

    RAY_LOG(DEBUG) << "Ray Start command: " << ray_start_cmd;
    RAY_CHECK(system(ray_start_cmd.c_str()) == 0);
    usleep(200 * 1000);
    return raylet_socket_name;
  }

  void StopRaylet(std::string raylet_socket_name) {
    std::string raylet_pid = raylet_socket_name + ".pid";
    std::string kill_9 = "kill -9 `cat " + raylet_pid + "`";
    RAY_LOG(DEBUG) << kill_9;
    ASSERT_TRUE(system(kill_9.c_str()) == 0);
    ASSERT_TRUE(system(("rm -rf " + raylet_socket_name).c_str()) == 0);
    ASSERT_TRUE(system(("rm -rf " + raylet_socket_name + ".pid").c_str()) == 0);
  }

  void SetUp() {}

  void TearDown() {}

  // Test tore provider.
  void TestStoreProvider(StoreProviderType type);

  // Test normal tasks.
  void TestNormalTask(const std::unordered_map<std::string, double> &resources);

  // Test actor tasks.
  void TestActorTask(const std::unordered_map<std::string, double> &resources,
                     bool is_direct_call);

  // Test actor failure case, verify that the tasks would either succeed or
  // fail with exceptions, in that case the return objects fetched from `Get`
  // contain errors.
  void TestActorFailure(const std::unordered_map<std::string, double> &resources,
                        bool is_direct_call);

  // Test actor failover case. Verify that actor can be reconstructed successfully,
  // and as long as we wait for actor reconstruction before submitting new tasks,
  // it is guaranteed that all tasks are successfully completed.
  void TestActorReconstruction(const std::unordered_map<std::string, double> &resources,
                               bool is_direct_call);

 protected:
  bool WaitForDirectCallActorState(CoreWorker &worker, const ActorID &actor_id,
                                   bool wait_alive, int timeout_ms);

  std::vector<std::string> raylet_socket_names_;
  std::vector<std::string> raylet_store_socket_names_;
  gcs::GcsClientOptions gcs_options_;
};

bool CoreWorkerTest::WaitForDirectCallActorState(CoreWorker &worker,
                                                 const ActorID &actor_id, bool wait_alive,
                                                 int timeout_ms) {
  auto condition_func = [&worker, actor_id, wait_alive]() -> bool {
    auto &task_submitters = worker.Tasks().task_submitters_;
    RAY_CHECK(task_submitters.count(TaskTransportType::DIRECT_ACTOR) > 0);
    auto submitter =
        worker.Tasks().task_submitters_[TaskTransportType::DIRECT_ACTOR].get();
    auto direct_actor_submitter =
        dynamic_cast<CoreWorkerDirectActorTaskSubmitter *>(submitter);
    RAY_CHECK(direct_actor_submitter != nullptr);
    bool actor_alive = direct_actor_submitter->IsActorAlive(actor_id);
    return wait_alive ? actor_alive : !actor_alive;
  };

  return WaitForCondition(condition_func, timeout_ms);
}

void CoreWorkerTest::TestNormalTask(
    const std::unordered_map<std::string, double> &resources) {
  CoreWorker driver(WorkerType::DRIVER, Language::PYTHON, raylet_store_socket_names_[0],
                    raylet_socket_names_[0], NextJobId(), gcs_options_, nullptr);

  // Test for tasks with by-value and by-ref args.
  {
    const int num_tasks = 100;
    for (int i = 0; i < num_tasks; i++) {
      auto buffer1 = GenerateRandomBuffer();
      auto buffer2 = GenerateRandomBuffer();

      ObjectID object_id;
      RAY_CHECK_OK(driver.Objects().Put(RayObject(buffer2, nullptr), &object_id));

      std::vector<TaskArg> args;
      args.emplace_back(TaskArg::PassByValue(buffer1));
      args.emplace_back(TaskArg::PassByReference(object_id));

      RayFunction func{ray::Language::PYTHON, {}};
      TaskOptions options;

      std::vector<ObjectID> return_ids;
      RAY_CHECK_OK(driver.Tasks().SubmitTask(func, args, options, &return_ids));

      ASSERT_EQ(return_ids.size(), 1);

      std::vector<std::shared_ptr<ray::RayObject>> results;
      RAY_CHECK_OK(driver.Objects().Get(return_ids, -1, &results));

      ASSERT_EQ(results.size(), 1);
      ASSERT_EQ(results[0]->GetData()->Size(), buffer1->Size() + buffer2->Size());
      ASSERT_EQ(memcmp(results[0]->GetData()->Data(), buffer1->Data(), buffer1->Size()),
                0);
      ASSERT_EQ(memcmp(results[0]->GetData()->Data() + buffer1->Size(), buffer2->Data(),
                       buffer2->Size()),
                0);
    }
  }
}

void CoreWorkerTest::TestActorTask(
    const std::unordered_map<std::string, double> &resources, bool is_direct_call) {
  CoreWorker driver(WorkerType::DRIVER, Language::PYTHON, raylet_store_socket_names_[0],
                    raylet_socket_names_[0], NextJobId(), gcs_options_, nullptr);

  auto actor_handle = CreateActorHelper(driver, resources, is_direct_call, 1000);

  // Test submitting some tasks with by-value args for that actor.
  {
    const int num_tasks = 100;
    for (int i = 0; i < num_tasks; i++) {
      auto buffer1 = GenerateRandomBuffer();
      auto buffer2 = GenerateRandomBuffer();

      // Create arguments with PassByRef and PassByValue.
      std::vector<TaskArg> args;
      args.emplace_back(TaskArg::PassByValue(buffer1));
      args.emplace_back(TaskArg::PassByValue(buffer2));

      TaskOptions options{1, resources};
      std::vector<ObjectID> return_ids;
      RayFunction func{ray::Language::PYTHON, {}};

      RAY_CHECK_OK(driver.Tasks().SubmitActorTask(*actor_handle, func, args, options,
                                                  &return_ids));
      ASSERT_EQ(return_ids.size(), 1);
      ASSERT_TRUE(return_ids[0].IsReturnObject());
      ASSERT_EQ(
          static_cast<TaskTransportType>(return_ids[0].GetTransportType()),
          is_direct_call ? TaskTransportType::DIRECT_ACTOR : TaskTransportType::RAYLET);

      std::vector<std::shared_ptr<ray::RayObject>> results;
      RAY_CHECK_OK(driver.Objects().Get(return_ids, -1, &results));

      ASSERT_EQ(results.size(), 1);
      ASSERT_EQ(results[0]->GetData()->Size(), buffer1->Size() + buffer2->Size());
      ASSERT_EQ(memcmp(results[0]->GetData()->Data(), buffer1->Data(), buffer1->Size()),
                0);
      ASSERT_EQ(memcmp(results[0]->GetData()->Data() + buffer1->Size(), buffer2->Data(),
                       buffer2->Size()),
                0);
    }
  }

  // Test submitting a task with both by-value and by-ref args for that actor.
  {
    uint8_t array1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t array2[] = {10, 11, 12, 13, 14, 15};

    auto buffer1 = std::make_shared<LocalMemoryBuffer>(array1, sizeof(array1));
    auto buffer2 = std::make_shared<LocalMemoryBuffer>(array2, sizeof(array2));

    ObjectID object_id;
    RAY_CHECK_OK(driver.Objects().Put(RayObject(buffer1, nullptr), &object_id));

    // Create arguments with PassByRef and PassByValue.
    std::vector<TaskArg> args;
    args.emplace_back(TaskArg::PassByReference(object_id));
    args.emplace_back(TaskArg::PassByValue(buffer2));

    TaskOptions options{1, resources};
    std::vector<ObjectID> return_ids;
    RayFunction func{ray::Language::PYTHON, {}};
    auto status =
        driver.Tasks().SubmitActorTask(*actor_handle, func, args, options, &return_ids);
    if (is_direct_call) {
      // For direct actor call, submitting a task with by-reference arguments
      // would fail.
      ASSERT_TRUE(!status.ok());
      return;
    }

    ASSERT_EQ(return_ids.size(), 1);

    std::vector<std::shared_ptr<ray::RayObject>> results;
    RAY_CHECK_OK(driver.Objects().Get(return_ids, -1, &results));

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0]->GetData()->Size(), buffer1->Size() + buffer2->Size());
    ASSERT_EQ(memcmp(results[0]->GetData()->Data(), buffer1->Data(), buffer1->Size()), 0);
    ASSERT_EQ(memcmp(results[0]->GetData()->Data() + buffer1->Size(), buffer2->Data(),
                     buffer2->Size()),
              0);
  }
}

void CoreWorkerTest::TestActorReconstruction(
    const std::unordered_map<std::string, double> &resources, bool is_direct_call) {
  CoreWorker driver(WorkerType::DRIVER, Language::PYTHON, raylet_store_socket_names_[0],
                    raylet_socket_names_[0], NextJobId(), gcs_options_, nullptr);

  // creating actor.
  auto actor_handle = CreateActorHelper(driver, resources, is_direct_call, 1000);

  // Wait for actor alive event.
  ASSERT_TRUE(WaitForDirectCallActorState(driver, actor_handle->ActorID(), true,
                                          30 * 1000 /* 30s */));
  RAY_LOG(INFO) << "actor has been created";

  // Test submitting some tasks with by-value args for that actor.
  {
    const int num_tasks = 100;
    const int task_index_to_kill_worker = (num_tasks + 1) / 2;
    std::vector<std::pair<ObjectID, std::vector<uint8_t>>> all_results;
    for (int i = 0; i < num_tasks; i++) {
      if (i == task_index_to_kill_worker) {
        RAY_LOG(INFO) << "killing worker";
        ASSERT_EQ(system("pkill mock_worker"), 0);

        // Wait for actor restruction event, and then for alive event.
        ASSERT_TRUE(WaitForDirectCallActorState(driver, actor_handle->ActorID(), false,
                                                30 * 1000 /* 30s */));
        ASSERT_TRUE(WaitForDirectCallActorState(driver, actor_handle->ActorID(), true,
                                                30 * 1000 /* 30s */));

        RAY_LOG(INFO) << "actor has been reconstructed";
      }

      // wait for actor being reconstructed.
      auto buffer1 = GenerateRandomBuffer();

      // Create arguments with PassByValue.
      std::vector<TaskArg> args;
      args.emplace_back(TaskArg::PassByValue(buffer1));

      TaskOptions options{1, resources};
      std::vector<ObjectID> return_ids;
      RayFunction func{ray::Language::PYTHON, {}};

      auto status =
          driver.Tasks().SubmitActorTask(*actor_handle, func, args, options, &return_ids);
      RAY_CHECK_OK(status);
      ASSERT_EQ(return_ids.size(), 1);
      // Verify if it's expected data.
      std::vector<std::shared_ptr<RayObject>> results;

      RAY_CHECK_OK(driver.Objects().Get(return_ids, -1, &results));
      ASSERT_EQ(results[0]->GetData()->Size(), buffer1->Size());
      ASSERT_EQ(*results[0]->GetData(), *buffer1);
    }
  }
}

void CoreWorkerTest::TestActorFailure(
    const std::unordered_map<std::string, double> &resources, bool is_direct_call) {
  CoreWorker driver(WorkerType::DRIVER, Language::PYTHON, raylet_store_socket_names_[0],
                    raylet_socket_names_[0], NextJobId(), gcs_options_, nullptr);

  // creating actor.
  auto actor_handle =
      CreateActorHelper(driver, resources, is_direct_call, 0 /* not reconstructable */);

  // Test submitting some tasks with by-value args for that actor.
  {
    const int num_tasks = 3000;
    const int task_index_to_kill_worker = (num_tasks + 1) / 2;
    std::vector<std::pair<ObjectID, std::shared_ptr<Buffer>>> all_results;
    for (int i = 0; i < num_tasks; i++) {
      if (i == task_index_to_kill_worker) {
        RAY_LOG(INFO) << "killing worker";
        ASSERT_EQ(system("pkill mock_worker"), 0);
      }

      // wait for actor being reconstructed.
      auto buffer1 = GenerateRandomBuffer();

      // Create arguments with PassByRef and PassByValue.
      std::vector<TaskArg> args;
      args.emplace_back(TaskArg::PassByValue(buffer1));

      TaskOptions options{1, resources};
      std::vector<ObjectID> return_ids;
      RayFunction func{ray::Language::PYTHON, {}};

      auto status =
          driver.Tasks().SubmitActorTask(*actor_handle, func, args, options, &return_ids);
      if (i < task_index_to_kill_worker) {
        RAY_CHECK_OK(status);
      }

      ASSERT_EQ(return_ids.size(), 1);
      all_results.emplace_back(std::make_pair(return_ids[0], buffer1));
    }

    for (int i = 0; i < num_tasks; i++) {
      const auto &entry = all_results[i];
      std::vector<ObjectID> return_ids;
      return_ids.push_back(entry.first);
      std::vector<std::shared_ptr<RayObject>> results;
      RAY_CHECK_OK(driver.Objects().Get(return_ids, -1, &results));
      ASSERT_EQ(results.size(), 1);

      if (results[0]->HasMetadata()) {
        // Verify if this is the desired error.
        std::string meta = std::to_string(static_cast<int>(rpc::ErrorType::ACTOR_DIED));
        ASSERT_TRUE(memcmp(results[0]->GetMetadata()->Data(), meta.data(), meta.size()) ==
                    0);
      } else {
        // Verify if it's expected data.
        ASSERT_EQ(*results[0]->GetData(), *entry.second);
      }
    }
  }
}

void CoreWorkerTest::TestStoreProvider(StoreProviderType type) {
  std::unique_ptr<CoreWorkerStoreProvider> provider_ptr;
  std::shared_ptr<CoreWorkerMemoryStore> memory_store;

  switch (type) {
  case StoreProviderType::LOCAL_PLASMA:
    provider_ptr = std::unique_ptr<CoreWorkerStoreProvider>(
        new CoreWorkerLocalPlasmaStoreProvider(raylet_store_socket_names_[0]));
    break;
  case StoreProviderType::MEMORY:
    memory_store = std::make_shared<CoreWorkerMemoryStore>();
    provider_ptr = std::unique_ptr<CoreWorkerStoreProvider>(
        new CoreWorkerMemoryStoreProvider(memory_store));
    break;
  default:
    RAY_LOG(FATAL) << "unspported store provider type " << static_cast<int>(type);
    break;
  }

  auto &provider = *provider_ptr;

  uint8_t array1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint8_t array2[] = {10, 11, 12, 13, 14, 15};

  std::vector<RayObject> buffers;
  buffers.emplace_back(std::make_shared<LocalMemoryBuffer>(array1, sizeof(array1)),
                       std::make_shared<LocalMemoryBuffer>(array1, sizeof(array1) / 2));
  buffers.emplace_back(std::make_shared<LocalMemoryBuffer>(array2, sizeof(array2)),
                       std::make_shared<LocalMemoryBuffer>(array2, sizeof(array2) / 2));

  std::vector<ObjectID> ids(buffers.size());
  for (size_t i = 0; i < ids.size(); i++) {
    ids[i] = ObjectID::FromRandom();
    RAY_CHECK_OK(provider.Put(buffers[i], ids[i]));
  }

  // Test Wait().
  std::vector<ObjectID> ids_with_duplicate;
  ids_with_duplicate.insert(ids_with_duplicate.end(), ids.begin(), ids.end());
  // add the same ids again to test `Get` with duplicate object ids.
  ids_with_duplicate.insert(ids_with_duplicate.end(), ids.begin(), ids.end());

  std::vector<ObjectID> wait_ids(ids_with_duplicate);
  ObjectID non_existent_id = ObjectID::FromRandom();
  wait_ids.push_back(non_existent_id);

  std::vector<bool> wait_results;
  RAY_CHECK_OK(provider.Wait(wait_ids, 5, 100, RandomTaskId(), &wait_results));
  ASSERT_EQ(wait_results.size(), 5);
  ASSERT_EQ(wait_results, std::vector<bool>({true, true, true, true, false}));

  // Test Get().
  std::vector<std::shared_ptr<RayObject>> results;
  RAY_CHECK_OK(provider.Get(ids_with_duplicate, -1, RandomTaskId(), &results));

  ASSERT_EQ(results.size(), ids_with_duplicate.size());
  for (size_t i = 0; i < ids_with_duplicate.size(); i++) {
    const auto &expected = buffers[i % ids.size()];
    ASSERT_EQ(results[i]->GetData()->Size(), expected.GetData()->Size());
    ASSERT_EQ(memcmp(results[i]->GetData()->Data(), expected.GetData()->Data(),
                     expected.GetData()->Size()),
              0);
    ASSERT_EQ(results[i]->GetMetadata()->Size(), expected.GetMetadata()->Size());
    ASSERT_EQ(memcmp(results[i]->GetMetadata()->Data(), expected.GetMetadata()->Data(),
                     expected.GetMetadata()->Size()),
              0);
  }

  // Test Delete().
  // clear the reference held.
  results.clear();

  RAY_CHECK_OK(provider.Delete(ids, true, false));

  usleep(200 * 1000);
  RAY_CHECK_OK(provider.Get(ids, 0, RandomTaskId(), &results));
  ASSERT_EQ(results.size(), 2);
  ASSERT_TRUE(!results[0]);
  ASSERT_TRUE(!results[1]);
}

class ZeroNodeTest : public CoreWorkerTest {
 public:
  ZeroNodeTest() : CoreWorkerTest(0) {}
};

class SingleNodeTest : public CoreWorkerTest {
 public:
  SingleNodeTest() : CoreWorkerTest(1) {}
};

class TwoNodeTest : public CoreWorkerTest {
 public:
  TwoNodeTest() : CoreWorkerTest(2) {}
};

TEST_F(SingleNodeTest, TestDirectActorTaskLocalFailure) {
  std::unordered_map<std::string, double> resources;
  TestActorFailure(resources, true);
}

}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  RAY_CHECK(argc == 4);
  ray::store_executable = std::string(argv[1]);
  ray::raylet_executable = std::string(argv[2]);
  ray::mock_worker_executable = std::string(argv[3]);
  return RUN_ALL_TESTS();
}
