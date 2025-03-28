<template>
  <div class="home">
    <h1>Agent Manager</h1>
    <AgentList />
    <AgentEvaluation />
    <RunAgent />
    <NewAgent />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import AgentList from '../components/AgentList.vue';
import AgentEvaluation from '../components/AgentEvaluation.vue';
import RunAgent from '../components/RunAgent.vue';
import NewAgent from '../components/NewAgent.vue';

export default defineComponent({
  name: 'HomeView',
  components: {
    AgentList,
    AgentEvaluation,
    RunAgent,
    NewAgent,
  },
});
</script>

<style scoped>
.home {
  padding: 20px;
}
</style>
