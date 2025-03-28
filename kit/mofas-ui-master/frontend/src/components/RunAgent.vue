<template>
  <div class="run-agent">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <h2>
            <el-icon><VideoPlay /></el-icon> Run Agent
          </h2>
        </div>
      </template>

      <el-form :model="form" :rules="rules" ref="formRef" label-width="120px">
        <el-form-item label="Agent Name" prop="agentName">
          <el-select
            v-model="form.agentName"
            placeholder="Select Agent"
            filterable
            style="width: 100%"
          >
            <el-option label="Reasoner" value="reasoner" />
            <el-option label="Solver" value="solver" />
            <el-option label="Planner" value="planner" />
            <el-option label="Executor" value="executor" />
          </el-select>
        </el-form-item>

        <el-form-item label="Parameters" prop="params">
          <el-input
            v-model="form.params"
            type="textarea"
            :rows="3"
            placeholder="Enter parameters (JSON format)"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="submitForm" :loading="loading">
            <el-icon><CaretRight /></el-icon> Run Agent
          </el-button>
          <el-button @click="resetForm">Reset</el-button>
        </el-form-item>
      </el-form>

      <div v-if="result" class="result-section">
        <el-divider content-position="left">Result</el-divider>
        <el-alert v-if="error" title="Error" type="error" :description="result" show-icon />
        <el-card v-else shadow="never" class="result-card">
          <pre>{{ result }}</pre>
        </el-card>
      </div>
    </el-card>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, reactive } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import type { FormInstance, FormRules } from 'element-plus'
import { VideoPlay, CaretRight } from '@element-plus/icons-vue'

export default defineComponent({
  name: 'RunAgent',
  components: {
    VideoPlay,
    CaretRight,
  },
  setup() {
    const formRef = ref<FormInstance>()
    const loading = ref(false)
    const error = ref(false)
    const result = ref('')

    const form = reactive({
      agentName: 'reasoner',
      params: '',
    })

    const rules = reactive<FormRules>({
      agentName: [{ required: true, message: 'Please select an agent', trigger: 'change' }],
      params: [
        {
          validator: (rule, value, callback) => {
            if (!value) {
              callback()
              return
            }
            try {
              JSON.parse(value)
              callback()
            } catch (e) {
              callback(new Error('Invalid JSON format'))
            }
          },
          trigger: 'blur',
        },
      ],
    })

    const submitForm = async () => {
      if (!formRef.value) return

      await formRef.value.validate(async (valid) => {
        if (valid) {
          loading.value = true
          error.value = false

          try {
            const params = form.params ? JSON.parse(form.params) : {}
            const response = await axios.post('/run-agent', {
              agentName: form.agentName,
              params,
            })
            result.value = JSON.stringify(response.data.result, null, 2)
            ElMessage.success('Agent executed successfully')
          } catch (err: any) {
            console.error('Run agent failed:', err)
            result.value = err.response?.data?.message || err.message || 'Unknown error occurred'
            error.value = true
            ElMessage.error('Failed to run agent')
          } finally {
            loading.value = false
          }
        }
      })
    }

    const resetForm = () => {
      if (!formRef.value) return
      formRef.value.resetFields()
      result.value = ''
      error.value = false
    }

    return {
      formRef,
      form,
      rules,
      loading,
      result,
      error,
      submitForm,
      resetForm,
    }
  },
})
</script>

<style scoped>
.run-agent {
  margin-bottom: 30px;
}
.card-header {
  display: flex;
  align-items: center;
}
.card-header h2 {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.2rem;
  color: #409eff;
}
.result-section {
  margin-top: 20px;
}
.result-card {
  background-color: #f8f8f8;
}
pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  margin: 0;
  padding: 10px;
  max-height: 300px;
  overflow-y: auto;
}
</style>
