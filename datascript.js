const io = require("socket.io-client");
require('dotenv').config();
const { createClient } = require("@supabase/supabase-js");
const http = require('http');

// --- HTTP Health Check for Cloud Hosting (Cloud Run / Render) ---
const port = process.env.PORT || 8080;
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Telemetry ingest service is running.\n');
}).listen(port, () => {
    console.log(`🌐 Health check server listening on port ${port}`);
});

// --- Supabase setup using credentials from .env ---
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
    throw new Error("SUPABASE_URL and SUPABASE_KEY must be set in your .env file!");
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Cache lookups to minimize DB queries for hierarchy IDs
const siteCache = new Map();
const deptCache = new Map();
const lineCache = new Map();

async function getOrCreateSite(siteName, gm, bu) {
    const name = siteName || "Default Site";
    if (siteCache.has(name)) return siteCache.get(name);

    let { data } = await supabase.from("sites").select("id").eq("site_name", name).maybeSingle();
    if (!data) {
        const { data: upserted, error: upsertErr } = await supabase
            .from("sites")
            .upsert([{ site_name: name, gm: gm || null, bu: bu || null }], { onConflict: "site_name" })
            .select("id")
            .single();
        if (upsertErr) {
            // Fallback select if upsert select had an issue
            const { data: fallback } = await supabase.from("sites").select("id").eq("site_name", name).single();
            data = fallback;
        } else {
            data = upserted;
        }
    }
    if (!data || !data.id) throw new Error(`Could not resolve site ID for ${name}`);
    siteCache.set(name, data.id);
    return data.id;
}

async function getOrCreateDepartment(siteId, deptName) {
    const name = deptName || "General";
    const cacheKey = `${siteId}:${name}`;
    if (deptCache.has(cacheKey)) return deptCache.get(cacheKey);

    let { data } = await supabase
        .from("departments")
        .select("id")
        .eq("site_id", siteId)
        .eq("department_name", name)
        .maybeSingle();

    if (!data) {
        const { data: inserted, error: insertErr } = await supabase
            .from("departments")
            .insert([{ site_id: siteId, department_name: name }])
            .select("id")
            .single();
        if (insertErr) {
            const { data: fallback } = await supabase
                .from("departments")
                .select("id")
                .eq("site_id", siteId)
                .eq("department_name", name)
                .single();
            data = fallback;
        } else {
            data = inserted;
        }
    }
    if (!data || !data.id) throw new Error(`Could not resolve department ID for ${name}`);
    deptCache.set(cacheKey, data.id);
    return data.id;
}

async function getOrCreateProductionLine(deptId, lineName) {
    const name = lineName || "Line 1";
    const cacheKey = `${deptId}:${name}`;
    if (lineCache.has(cacheKey)) return lineCache.get(cacheKey);

    let { data } = await supabase
        .from("production_lines")
        .select("id")
        .eq("department_id", deptId)
        .eq("line_name", name)
        .maybeSingle();

    if (!data) {
        const { data: inserted, error: insertErr } = await supabase
            .from("production_lines")
            .insert([{ department_id: deptId, line_name: name }])
            .select("id")
            .single();
        if (insertErr) {
            const { data: fallback } = await supabase
                .from("production_lines")
                .select("id")
                .eq("department_id", deptId)
                .eq("line_name", name)
                .single();
            data = fallback;
        } else {
            data = inserted;
        }
    }
    if (!data || !data.id) throw new Error(`Could not resolve line ID for ${name}`);
    lineCache.set(cacheKey, data.id);
    return data.id;
}

// --- Connect to WebSocket ---
const socket = io.connect("https://virtualfactory.online:3000");

socket.on("connect", () => {
    console.log("Connected to virtualfactory.online WebSocket");
});

let lastInsertedTime = 0;
const TEN_MINUTES_MS = 10 * 60 * 1000;

socket.on("update", async (data) => {
    const now = Date.now();
    if (now - lastInsertedTime < TEN_MINUTES_MS) {
        // Skip processing until 10 minutes have elapsed
        return;
    }
    lastInsertedTime = now;
    console.log("📥 Received WebSocket update cycle (Inserting data for 10-minute interval)...");

    try {
        const recordTimestamp = data.timestamp ? new Date(data.timestamp) : new Date();
        const payload = data;

        if (payload.Enterprise) {
            const sites = ["Dallas", "Austin", "Smithfield", "Site"];
            for (const siteKey of sites) {
                if (payload.Enterprise[siteKey]) {
                    const siteObj = payload.Enterprise[siteKey] || {};
                    let siteName = siteObj.Location?.value || siteKey;
                    if (siteName === "Dallas") siteName = "Biyagama";
                    if (siteName === "Austin") siteName = "Katunayake";

                    const gm = siteObj.GM?.value || null;
                    const bu = siteObj.BU?.value || null;

                    const siteId = await getOrCreateSite(siteName, gm, bu);

                    const areas = siteKey === "Smithfield" ? ["Test"] : siteKey === "Site" ? ["Area"] : ["Press", "Heat Treat", "Assembly"];
                    for (const areaKey of areas) {
                        if (siteObj[areaKey]) {
                            const areaObj = siteObj[areaKey] || {};
                            const deptId = await getOrCreateDepartment(siteId, areaKey);

                            for (const lineKey in areaObj) {
                                const lineObj = areaObj[lineKey] || {};
                                const lineId = await getOrCreateProductionLine(deptId, lineKey);

                                // 1. ERP Orders
                                if (lineObj.ERP) {
                                    const erp = lineObj.ERP;
                                    await supabase.from("erp_orders").insert([{
                                        line_id: lineId,
                                        order_number: erp.OrderNumber?.value || null,
                                        order_status: erp.OrderStatus?.value || null,
                                        item_number: erp.ItemNumber?.value || null,
                                        item_description: erp.ItemDescription?.value || null,
                                        ordered_quantity: erp.OrderedQuantity?.value || null,
                                        produced_quantity: erp.ProducedQuantity?.value || null,
                                        remaining_quantity: erp.RemainingQuantity?.value || null,
                                        available_quantity: erp.AvailableQuantity?.value || null,
                                        reserved_quantity: erp.ReservedQuantity?.value || null,
                                        scheduled_start_time: erp.ScheduledStartTime?.value ? new Date(erp.ScheduledStartTime.value) : null,
                                        scheduled_end_time: erp.ScheduledEndTime?.value ? new Date(erp.ScheduledEndTime.value) : null,
                                        actual_start_time: erp.ActualStartTime?.value ? new Date(erp.ActualStartTime.value) : null,
                                        actual_end_time: erp.ActualEndTime?.value ? new Date(erp.ActualEndTime.value) : null,
                                        bom: erp.BOM?.value || null,
                                        location: erp.Location?.value || null,
                                        timestamp: recordTimestamp
                                    }]);
                                }

                                // 2. KPI Metrics (MES -> kpi_metrics)
                                if (lineObj.MES && lineObj.MES.KPIs) {
                                    const kpi = lineObj.MES.KPIs;
                                    await supabase.from("kpi_metrics").insert([{
                                        line_id: lineId,
                                        source: "MES",
                                        oee: kpi.OEE?.value || null,
                                        availability: kpi.Availability?.value || null,
                                        performance: kpi.Performance?.value || null,
                                        quality: kpi.Quality?.value || null,
                                        teep: kpi.TEEP?.value || null,
                                        mtbf: kpi.MTBF?.value || null,
                                        mttr: kpi.MTTR?.value || null,
                                        timestamp: kpi.OEE?.timestamp ? new Date(kpi.OEE.timestamp) : recordTimestamp
                                    }]);
                                }

                                // 3. Quality Inspections (Quality -> quality_inspections)
                                if (lineObj.MES && lineObj.MES.Quality) {
                                    const qc = lineObj.MES.Quality;
                                    await supabase.from("quality_inspections").insert([{
                                        line_id: lineId,
                                        source: "MES",
                                        order_number: qc.OrderNumber?.value || null,
                                        item_number: qc.ItemNumber?.value || null,
                                        inspection_result: qc.InspectionResult?.value || null,
                                        rejection_reason: qc.RejectionReason?.value || null,
                                        accepted_quantity: qc.AcceptedQuantity?.value || null,
                                        rejection_quantity: qc.RejectionQuantity?.value || null,
                                        timestamp: qc.OrderNumber?.timestamp ? new Date(qc.OrderNumber.timestamp) : recordTimestamp
                                    }]);
                                }

                                // 4. Maintenance Records
                                if (lineObj.MES && lineObj.MES.Maintenance) {
                                    const m = lineObj.MES.Maintenance;
                                    await supabase.from("maintenance_records").insert([{
                                        line_id: lineId,
                                        source: "MES",
                                        machine_id: m.MachineID?.value || null,
                                        maintenance_status: m.MaintenanceStatus?.value || null,
                                        last_maintenance_date: m.LastMaintenanceDate?.value ? new Date(m.LastMaintenanceDate.value) : null,
                                        next_maintenance_date: m.NextMaintenanceDate?.value ? new Date(m.NextMaintenanceDate.value) : null,
                                        maintenance_history: m.MaintenanceHistory?.value || null,
                                        timestamp: m.MachineID?.timestamp ? new Date(m.MachineID.timestamp) : recordTimestamp
                                    }]);
                                }

                                // 5. S88 Batch Control
                                if (lineObj.S88 && lineObj.S88.value) {
                                    let s88Data = {};
                                    try {
                                        s88Data = typeof lineObj.S88.value === "string" ? JSON.parse(lineObj.S88.value) : lineObj.S88.value;
                                    } catch (e) {
                                        s88Data = {};
                                    }
                                    if (s88Data.BatchControl) {
                                        const eq = s88Data.BatchControl.EquipmentModule || {};
                                        const cm = s88Data.BatchControl.ControlModule || {};
                                        const rm = s88Data.BatchControl.RecipeManagement || {};
                                        const hmi = s88Data.BatchControl.HMI || {};
                                        const dc = s88Data.BatchControl.DataCollection || {};
                                        const sc = s88Data.BatchControl.SafetyCompliance || {};
                                        await supabase.from("s88_batch_control").insert([{
                                            line_id: lineId,
                                            source: "S88",
                                            batch_mixing_tank_status: eq.BatchMixingTankStatus || null,
                                            bottler_status: eq.BottlerStatus || null,
                                            capper_status: eq.CapperStatus || null,
                                            temperature_controller: cm.TemperatureController || null,
                                            volume_control: cm.VolumeControl || null,
                                            soda_recipe: rm.SodaRecipe || null,
                                            production_parameters: rm.ProductionParameters || null,
                                            operator_interface_status: hmi.OperatorInterfaceStatus || null,
                                            process_data: dc.ProcessData || null,
                                            quality_data: dc.QualityData || null,
                                            safety_status: sc.SafetyStatus || null,
                                            timestamp: lineObj.S88.timestamp ? new Date(lineObj.S88.timestamp) : recordTimestamp
                                        }]);
                                    }
                                }

                                // 6. Process Variables (Edge -> process_variables)
                                if (lineObj.Edge) {
                                    const edge = lineObj.Edge;
                                    await supabase.from("process_variables").insert([{
                                        line_id: lineId,
                                        source: "Edge",
                                        state: edge.State?.value || null,
                                        waste: edge.Waste?.value || null,
                                        infeed: edge.Infeed?.value || null,
                                        outfeed: edge.Outfeed?.value || null,
                                        spindle_speed: edge.Process?.SpindleSpeed?.value || null,
                                        feed_rate: edge.Process?.FeedRate?.value || null,
                                        tool_wear: edge.Process?.ToolWear?.value || null,
                                        coolant_temperature: edge.Process?.CoolantTemperature?.value || null,
                                        vibration: edge.Process?.Vibration?.value || null,
                                        power_consumption: edge.Process?.PowerConsumption?.value || null,
                                        tool_change_count: edge.Process?.ToolChangeCount?.value || null,
                                        material_temperature: edge.Process?.MaterialTemperature?.value || null,
                                        part_dimensions: edge.Process?.PartDimensions?.value || null,
                                        surface_finish: edge.Process?.SurfaceFinish?.value || null,
                                        timestamp: edge.State?.timestamp ? new Date(edge.State.timestamp) : recordTimestamp
                                    }]);
                                }

                                // 7. ISO 55001 Metrics (55001 -> iso55001_metrics)
                                if (lineObj["55001"] && lineObj["55001"].value) {
                                    let isoData = {};
                                    try {
                                        isoData = typeof lineObj["55001"].value === "string" ? JSON.parse(lineObj["55001"].value) : lineObj["55001"].value;
                                    } catch (e) {
                                        isoData = {};
                                    }
                                    if (isoData.AssetLifecycle) {
                                        await supabase.from("iso55001_metrics").insert([{
                                            line_id: lineId,
                                            status: isoData.AssetLifecycle.Status || null,
                                            maintenance_schedule: isoData.AssetLifecycle.MaintenanceSchedule ? new Date(isoData.AssetLifecycle.MaintenanceSchedule) : null,
                                            risk_level: isoData.RiskManagement?.RiskLevel || null,
                                            mitigation_plan: isoData.RiskManagement?.MitigationPlan || null,
                                            oee: isoData.PerformanceIndicators?.OEE || null,
                                            mtbf: isoData.PerformanceIndicators?.MTBF || null,
                                            mttr: isoData.PerformanceIndicators?.MTTR || null,
                                            regulatory_status: isoData.Compliance?.RegulatoryStatus || null,
                                            last_review_date: isoData.ContinuousImprovement?.LastReviewDate ? new Date(isoData.ContinuousImprovement.LastReviewDate) : null,
                                            planned_action: isoData.ContinuousImprovement?.PlannedAction || null,
                                            timestamp: lineObj["55001"].timestamp ? new Date(lineObj["55001"].timestamp) : recordTimestamp
                                        }]);
                                    }
                                }

                                // 8. Dashboard Status (Dashboard -> dashboard_status)
                                if (lineObj.Dashboard && lineObj.Dashboard.value) {
                                    let dashData = {};
                                    try {
                                        dashData = typeof lineObj.Dashboard.value === "string" ? JSON.parse(lineObj.Dashboard.value) : lineObj.Dashboard.value;
                                    } catch (e) {
                                        dashData = {};
                                    }
                                    await supabase.from("dashboard_status").insert([{
                                        line_id: lineId,
                                        oee: dashData.OEE || null,
                                        availability: dashData.Availability || null,
                                        performance: dashData.Performance || null,
                                        quality: dashData.Quality || null,
                                        current_batch_status: dashData.CurrentBatchStatus || null,
                                        maintenance_status: dashData.MaintenanceStatus || null,
                                        timestamp: lineObj.Dashboard.timestamp ? new Date(lineObj.Dashboard.timestamp) : recordTimestamp
                                    }]);
                                }

                            }
                        }
                    }
                }
            }
        }
        console.log("✅ Successfully inserted telemetry cycle into chatbot tables!");
    } catch (err) {
        console.error("❌ Error inserting data:", err);
    }
});

socket.on("disconnect", () => {
    console.log("Disconnected from WebSocket");
});
