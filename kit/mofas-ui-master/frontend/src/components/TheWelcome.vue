<script setup lang="ts">
import DocumentationIcon from './icons/IconDocumentation.vue'
import ToolingIcon from './icons/IconTooling.vue'
import EcosystemIcon from './icons/IconEcosystem.vue'
import CommunityIcon from './icons/IconCommunity.vue'
import SupportIcon from './icons/IconSupport.vue'
import AgentList from './AgentList.vue'
import NewAgent from './NewAgent.vue'
import RunAgent from './RunAgent.vue'

const openReadmeInEditor = () => fetch('/__open-in-editor?file=README.md')
</script>

<template>
  <div class="cli-dashboard">
    <div class="dashboard-header">
      <h1>CLI工具管理平台</h1>
      <p class="subtitle">简化命令行工具使用，提升工作效率</p>
    </div>

    <div class="dashboard-main">
      <!-- 左侧导航 -->
      <div class="sidebar">
        <div class="nav-item active">
          <DocumentationIcon class="nav-icon" />
          <span>仪表盘</span>
        </div>
        <div class="nav-item">
          <ToolingIcon class="nav-icon" />
          <span>CLI工具</span>
        </div>
        <div class="nav-item">
          <EcosystemIcon class="nav-icon" />
          <span>执行历史</span>
        </div>
        <div class="nav-item">
          <CommunityIcon class="nav-icon" />
          <span>配置中心</span>
        </div>
        <div class="nav-item">
          <SupportIcon class="nav-icon" />
          <span>帮助文档</span>
        </div>
      </div>

      <!-- 主内容区 -->
      <div class="content-area">
        <!-- Agent管理区域 -->
        <div class="agent-management">
          <h2>CLI代理管理</h2>
          <div class="agent-cards">
            <div class="agent-card">
              <AgentList />
            </div>
            <div class="agent-card">
              <NewAgent />
            </div>
          </div>
        </div>

        <!-- 快速操作区域 -->
        <div class="quick-actions">
          <h2>快速操作</h2>
          <div class="action-cards">
            <div class="action-card">
              <RunAgent />
            </div>
            <div class="action-card">
              <h3>常用命令</h3>
              <ul class="command-list">
                <li><code>agent init</code> - 初始化新代理</li>
                <li><code>agent run</code> - 运行指定代理</li>
                <li><code>agent config</code> - 配置代理参数</li>
                <li><code>agent update</code> - 更新代理程序</li>
              </ul>
              <a href="javascript:void(0)" @click="openReadmeInEditor" class="doc-link"
                >查看完整文档</a
              >
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cli-dashboard {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.dashboard-header {
  text-align: center;
  margin-bottom: 30px;
}

.subtitle {
  color: #666;
  font-size: 16px;
}

.dashboard-main {
  display: flex;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
  min-height: 600px;
}

.sidebar {
  width: 220px;
  background-color: #f8f9fa;
  padding: 20px 0;
  border-right: 1px solid #eee;
}

.nav-item {
  padding: 12px 20px;
  display: flex;
  align-items: center;
  cursor: pointer;
  transition: all 0.3s;
}

.nav-item:hover {
  background-color: #e9ecef;
}

.nav-item.active {
  background-color: #e9ecef;
  font-weight: bold;
  border-left: 3px solid #42b983;
}

.nav-icon {
  width: 20px;
  height: 20px;
  margin-right: 10px;
}

.content-area {
  flex: 1;
  padding: 25px;
  background-color: #fff;
}

.agent-management,
.quick-actions {
  margin-bottom: 30px;
}

.agent-cards,
.action-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 15px;
}

.agent-card,
.action-card {
  background-color: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
  transition:
    transform 0.3s,
    box-shadow 0.3s;
}

.agent-card:hover,
.action-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

h2 {
  font-size: 18px;
  margin-bottom: 15px;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

h3 {
  font-size: 16px;
  margin-bottom: 10px;
  color: #333;
}

.command-list {
  list-style: none;
  padding: 0;
}

.command-list li {
  padding: 8px 0;
  border-bottom: 1px dashed #eee;
}

.command-list code {
  background-color: #eee;
  padding: 3px 6px;
  border-radius: 4px;
  font-family: monospace;
}

.doc-link {
  display: inline-block;
  margin-top: 15px;
  color: #42b983;
  text-decoration: none;
}

.doc-link:hover {
  text-decoration: underline;
}

@media (max-width: 768px) {
  .dashboard-main {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid #eee;
    padding: 10px 0;
  }

  .nav-item {
    padding: 8px 15px;
  }
}
</style>
