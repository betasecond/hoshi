<template>
  <div class="about">
    <el-row :gutter="20" justify="center">
      <el-col :span="18" :lg="16">
        <el-card shadow="hover" class="about-card">
          <template #header>
            <div class="about-header">
              <h1>About Mofas UI Agent Manager</h1>
            </div>
          </template>

          <el-descriptions title="Project Information" :column="1" border>
            <el-descriptions-item label="Project Name">Mofas UI</el-descriptions-item>
            <el-descriptions-item label="Version">1.0.0</el-descriptions-item>
            <el-descriptions-item label="Framework">Vue 3 + TypeScript</el-descriptions-item>
            <el-descriptions-item label="UI Library">Element Plus</el-descriptions-item>
            <el-descriptions-item label="Description">
              A powerful interface for managing and running AI agents with capabilities for
              evaluation and monitoring.
            </el-descriptions-item>
          </el-descriptions>

          <h3 class="mt-4">Features</h3>
          <el-row :gutter="20" class="feature-cards">
            <el-col :span="12" :sm="8" :md="6" v-for="(feature, index) in features" :key="index">
              <el-card shadow="hover" class="feature-card">
                <el-icon :size="32" class="feature-icon">
                  <component :is="feature.icon"></component>
                </el-icon>
                <h4>{{ feature.title }}</h4>
                <p>{{ feature.description }}</p>
              </el-card>
            </el-col>
          </el-row>

          <div class="footer-info">
            <p>© 2025 Hoshi Kit - All Rights Reserved</p>
            <el-link type="primary" href="https://github.com/example/mofas-ui" target="_blank">
              <el-icon><Link /></el-icon> GitHub Repository
            </el-link>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'
import {
  Monitor,
  Connection,
  DataAnalysis,
  Cpu,
  Link,
  Timer,
  Setting,
  Help,
} from '@element-plus/icons-vue'

export default defineComponent({
  name: 'AboutView',
  components: {
    Monitor,
    Connection,
    DataAnalysis,
    Cpu,
    Link,
    Timer,
    Setting,
    Help,
  },
  setup() {
    const features = [
      {
        title: 'Agent Management',
        description: 'Create, update, and delete agent configurations',
        icon: 'Setting',
      },
      {
        title: 'Agent Execution',
        description: 'Run agents and view results in real-time',
        icon: 'VideoPlay',
      },
      {
        title: 'Performance Metrics',
        description: 'Track and analyze agent performance',
        icon: 'DataAnalysis',
      },
      {
        title: 'Resource Monitoring',
        description: 'Monitor system resources used by agents',
        icon: 'Monitor',
      },
      {
        title: 'API Integration',
        description: 'Connect with external services and APIs',
        icon: 'Connection',
      },
      {
        title: 'Batch Processing',
        description: 'Run multiple agents in sequence or parallel',
        icon: 'Cpu',
      },
      {
        title: 'Scheduling',
        description: 'Schedule agent runs at specific times',
        icon: 'Timer',
      },
      {
        title: 'Documentation',
        description: 'Comprehensive guides and API documentation',
        icon: 'Help',
      },
    ]

    return { features }
  },
})
</script>

<style>
.about {
  padding: 30px;
  min-height: 100vh;
  background-color: #f5f7fa;
}
.about-card {
  margin-bottom: 20px;
}
.about-header {
  text-align: center;
}
.about-header h1 {
  margin: 0;
  color: #409eff;
}
.mt-4 {
  margin-top: 20px;
}
.feature-cards {
  margin-top: 20px;
  margin-bottom: 30px;
}
.feature-card {
  height: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  margin-bottom: 20px;
  transition: transform 0.3s;
}
.feature-card:hover {
  transform: translateY(-5px);
}
.feature-icon {
  margin: 10px 0;
  color: #409eff;
}
.feature-card h4 {
  margin: 10px 0;
  color: #303133;
}
.feature-card p {
  font-size: 0.9rem;
  color: #606266;
}
.footer-info {
  margin-top: 30px;
  text-align: center;
  color: #909399;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}
@media (max-width: 768px) {
  .about {
    padding: 15px;
  }
}
</style>
