{{/*
Expand the name of the chart.
*/}}
{{- define "aurelius.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "aurelius.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aurelius.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aurelius.labels" -}}
helm.sh/chart: {{ include "aurelius.chart" . }}
{{ include "aurelius.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aurelius.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aurelius.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "aurelius.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "aurelius.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Gateway labels
*/}}
{{- define "aurelius.gateway.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: gateway
{{- end }}

{{/*
Gateway selector labels
*/}}
{{- define "aurelius.gateway.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: gateway
{{- end }}

{{/*
DICOM service labels
*/}}
{{- define "aurelius.dicom.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: dicom-service
{{- end }}

{{/*
DICOM service selector labels
*/}}
{{- define "aurelius.dicom.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: dicom-service
{{- end }}

{{/*
ML service labels
*/}}
{{- define "aurelius.ml.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: ml-service
{{- end }}

{{/*
ML service selector labels
*/}}
{{- define "aurelius.ml.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: ml-service
{{- end }}

{{/*
Render service labels
*/}}
{{- define "aurelius.render.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: render-service
{{- end }}

{{/*
Render service selector labels
*/}}
{{- define "aurelius.render.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: render-service
{{- end }}

{{/*
Annotation service labels
*/}}
{{- define "aurelius.annotation.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: annotation-service
{{- end }}

{{/*
Annotation service selector labels
*/}}
{{- define "aurelius.annotation.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: annotation-service
{{- end }}

{{/*
Worklist service labels
*/}}
{{- define "aurelius.worklist.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: worklist-service
{{- end }}

{{/*
Worklist service selector labels
*/}}
{{- define "aurelius.worklist.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: worklist-service
{{- end }}

{{/*
Celery worker labels
*/}}
{{- define "aurelius.celery.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: celery-worker
{{- end }}

{{/*
Celery worker selector labels
*/}}
{{- define "aurelius.celery.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: celery-worker
{{- end }}

{{/*
Web UI labels
*/}}
{{- define "aurelius.webui.labels" -}}
{{ include "aurelius.labels" . }}
app.kubernetes.io/component: web-ui
{{- end }}

{{/*
Web UI selector labels
*/}}
{{- define "aurelius.webui.selectorLabels" -}}
{{ include "aurelius.selectorLabels" . }}
app.kubernetes.io/component: web-ui
{{- end }}

{{/*
Return the appropriate apiVersion for HPA
*/}}
{{- define "aurelius.hpa.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "autoscaling/v2" }}
{{- print "autoscaling/v2" }}
{{- else }}
{{- print "autoscaling/v2beta2" }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for PodDisruptionBudget
*/}}
{{- define "aurelius.pdb.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "policy/v1" }}
{{- print "policy/v1" }}
{{- else }}
{{- print "policy/v1beta1" }}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL host
*/}}
{{- define "aurelius.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" .Release.Name }}
{{- else }}
{{- .Values.externalDatabase.host }}
{{- end }}
{{- end }}

{{/*
Return the Redis host
*/}}
{{- define "aurelius.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" .Release.Name }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Return the MinIO host
*/}}
{{- define "aurelius.minio.host" -}}
{{- if .Values.minio.enabled }}
{{- printf "%s-minio" .Release.Name }}
{{- else }}
{{- .Values.externalS3.endpoint }}
{{- end }}
{{- end }}
