--[[
  Orthanc Lua Hook for Cancer AI Integration

  This script automatically triggers cancer AI analysis when:
  1. A new DICOM study is fully received in Orthanc
  2. The study is from specific modalities (CT, MRI, X-Ray)

  It sends a request to the Cancer AI service via the Gateway API.
]]--

-- Configuration
local GATEWAY_URL = os.getenv("GATEWAY_URL") or "http://gateway:8000"
local CANCER_AI_ENDPOINT = GATEWAY_URL .. "/cancer-ai/predict/dicom"
local AUTO_ANALYZE_MODALITIES = {"CT", "MR", "CR", "DX", "MG"}  -- CT, MRI, X-Ray, Digital Radiography, Mammography

-- Helper function to check if value is in table
function table_contains(table, value)
    for _, v in ipairs(table) do
        if v == value then
            return true
        end
    end
    return false
end

-- Helper function to make HTTP POST request
function http_post(url, body)
    local command = string.format(
        'curl -X POST -H "Content-Type: application/json" -d \'%s\' %s',
        body, url
    )
    os.execute(command)
end

-- Main hook: Called when a study is stable (all instances received)
function OnStableStudy(studyId, tags, metadata)
    -- Extract study information
    local studyInstanceUID = tags["StudyInstanceUID"]
    local modality = tags["Modality"] or ""
    local patientAge = tags["PatientAge"] or "0"
    local studyDescription = tags["StudyDescription"] or ""

    print(string.format(
        "[Cancer AI Hook] Study received: %s (Modality: %s, Age: %s)",
        studyInstanceUID, modality, patientAge
    ))

    -- Check if modality is in auto-analyze list
    if table_contains(AUTO_ANALYZE_MODALITIES, modality) then
        print(string.format(
            "[Cancer AI Hook] Auto-analyzing study %s with modality %s",
            studyInstanceUID, modality
        ))

        -- Prepare request body
        local requestBody = string.format([[{
            "study_instance_uid": "%s",
            "modality": "%s",
            "patient_age": "%s",
            "study_description": "%s",
            "auto_triggered": true
        }]], studyInstanceUID, modality, patientAge, studyDescription)

        -- Send async request to Cancer AI (fire and forget)
        -- In production, this should queue a job instead
        http_post(CANCER_AI_ENDPOINT, requestBody)

        print(string.format(
            "[Cancer AI Hook] Triggered analysis for study %s",
            studyInstanceUID
        ))
    else
        print(string.format(
            "[Cancer AI Hook] Skipping study %s - modality %s not in auto-analyze list",
            studyInstanceUID, modality
        ))
    end
end

-- Hook for new series (optional - for early analysis)
function OnStableSeries(seriesId, tags, metadata)
    -- This could be used for real-time analysis as series arrive
    -- Currently disabled to avoid redundant processing
    -- Uncomment if you want per-series analysis

    -- local seriesInstanceUID = tags["SeriesInstanceUID"]
    -- local modality = tags["Modality"] or ""
    -- print(string.format("[Cancer AI Hook] Series stable: %s (Modality: %s)", seriesInstanceUID, modality))
end

-- Logging startup
print("[Cancer AI Hook] Lua script loaded successfully")
print(string.format("[Cancer AI Hook] Gateway URL: %s", GATEWAY_URL))
print(string.format("[Cancer AI Hook] Auto-analyze modalities: %s", table.concat(AUTO_ANALYZE_MODALITIES, ", ")))
