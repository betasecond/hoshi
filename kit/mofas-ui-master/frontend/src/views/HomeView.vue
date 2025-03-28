<template>
  <div class="home">
    <el-page-header :icon="HomeFilled" title="Agent Manager Dashboard" />

    <el-tabs type="border-card" class="main-tabs">
      <el-tab-pane>
        <template #label>
          <el-icon><Menu /></el-icon>
          <span>Agent List</span>
        </template>
        <AgentList />
      </el-tab-pane>

      <el-tab-pane>
        <template #label>
          <el-icon><VideoPlay /></el-icon>
          <span>Run Agent</span>
        </template>
        <RunAgent />
      </el-tab-pane>

      <el-tab-pane>
        <template #label>
          <el-icon><Histogram /></el-icon>
          <span>Evaluation</span>
        </template>
        <AgentEvaluation />
      </el-tab-pane>

      <el-tab-pane>
        <template #label>
          <el-icon><Plus /></el-icon>
          <span>Add New Agent</span>
        </template>
        <NewAgent />
      </el-tab-pane>
    </el-tabs>

    <el-backtop :right="20" :bottom="20" />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import AgentList from '../components/AgentList.vue'
import AgentEvaluation from '../components/AgentEvaluation.vue'
import RunAgent from '../components/RunAgent.vue'
import NewAgent from '../components/NewAgent.vue'
import { Menu, VideoPlay, Histogram, Plus, HomeFilled } from '@element-plus/icons-vue'

export default defineComponent({
  name: 'HomeView',
  components: {
    AgentList,
    AgentEvaluation,
    RunAgent,
    NewAgent,
    Menu,
    VideoPlay,
    Histogram,
    Plus,
    HomeFilled,
  },
})
</script>

<style scoped>
.home {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}
.main-tabs {
  margin-top: 20px;
}
:deep(.el-tabs__item) {
  display: flex;
  align-items: center;
}
:deep(.el-tabs__item .el-icon) {
  margin-right: 5px;
}
:deep(.el-page-header__title) {
  font-size: 24px;
  font-weight: bold;
}
</style>
