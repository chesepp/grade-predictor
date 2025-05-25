#include <iostream>
#include <implot.h>
#include <imgui.h>
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <imgui_impl_win32.h>
#include <tchar.h>
#include <Windows.h>
#include <vector>
#include <cmath>
#include <algorithm>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

float predict(float x, const std::vector<float>& a) {
    float y_pred = 0.0f;
    float x_pow = 1.0f;
    for (auto coeff : a) {
        y_pred += coeff * x_pow;
        x_pow *= x;
    }
    return y_pred;
}

// Perform gradient descent to fit polynomial coefficients
void polynomial_regression_gd(
    const std::vector<float>& x_data,
    const std::vector<float>& y_data,
    std::vector<float>& a,
    float learning_rate = 0.001f,
    int iterations = 100000)
{
    int N = (int)x_data.size();
    int degree = (int)a.size() - 1;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<float> gradients(degree + 1, 0.0f);

        // Calculate gradients for all coefficients
        for (int i = 0; i < N; ++i) {
            float y_pred = predict(x_data[i], a);
            float error = y_data[i] - y_pred;

            float x_pow = 1.0f;
            for (int j = 0; j <= degree; ++j) {
                gradients[j] += -2 * error * x_pow;
                x_pow *= x_data[i];
            }
        }

        // Average gradients and update coefficients
        for (int j = 0; j <= degree; ++j) {
            gradients[j] /= N;
            a[j] -= learning_rate * gradients[j];
        }

        // Optional: print loss every 1000 iterations
        if (iter % 1000 == 0) {
            float loss = 0.0f;
            for (int i = 0; i < N; ++i) {
                float e = y_data[i] - predict(x_data[i], a);
                loss += e * e;
            }
            loss /= N;
            std::cout << "Iteration " << iter << ", Loss: " << loss << "\n";
        }
    }
}

ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;
HWND hwnd = nullptr;


LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}
void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}
void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

bool CreateDeviceD3D(HWND hwnd)
{
    if (!hwnd) {
        MessageBoxA(0, "Invalid HWND passed to CreateDeviceD3D!", "Error", MB_OK);
        return false;
    }
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;              // Use automatic sizing
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;              // No multi-sampling
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[] = {
    D3D_FEATURE_LEVEL_11_1,
    D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_10_0,
    D3D_FEATURE_LEVEL_9_3,
    D3D_FEATURE_LEVEL_9_2,
    D3D_FEATURE_LEVEL_9_1
    };

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createDeviceFlags,
        featureLevelArray,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        &featureLevel,
        &g_pd3dDeviceContext);

    if (FAILED(hr)) {
        char msg[128];
        sprintf_s(msg, "D3D11CreateDeviceAndSwapChain failed. HRESULT: 0x%08X", hr);
        MessageBoxA(0, msg, "D3D11 Error", MB_OK | MB_ICONERROR);
        return false;
    }
    if (!g_pd3dDevice || !g_pd3dDeviceContext) {
        MessageBoxA(0, "D3D11 device/context not initialized!", "Fatal Error", MB_OK | MB_ICONERROR);
        return 1; // or exit gracefully
    }
    // Create render target view
    CreateRenderTarget();
    return true;
}
void CreateConsole()
{
    AllocConsole();
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
    freopen_s(&fp, "CONIN$", "r", stdin);
    std::cout.clear();
    std::cerr.clear();
    std::cin.clear();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    CreateConsole();
    WNDCLASSEX wc = { sizeof(WNDCLASSEX) };
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;          // Your window procedure
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszMenuName = nullptr;
    wc.lpszClassName = L"MyWindowClass";
    wc.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);

    if (!RegisterClassEx(&wc))
    {
        MessageBox(nullptr, L"Failed to register window class", L"Error", MB_OK | MB_ICONERROR);
        return -1;
    }

    HWND hwnd = CreateWindowEx(0, L"MyWindowClass", L"ImGuiWindow", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1280, 720, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    if (hwnd == nullptr)
    {
        MessageBox(nullptr, L"Failed to create window", L"Error", MB_OK | MB_ICONERROR);
    }
    if (!CreateDeviceD3D(hwnd)) {
        return -1;
    }
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    MSG msg = {};
    ZeroMemory(&msg, sizeof(msg));

    std::vector<float> x_data = { 1, 2, 3, 4, 5, 6,7,8,9,10 };
    std::vector<float> y_data = { 75, 85, 90, 95, 88,90,86,89,90,86 };
    std::vector<float> x_data_normalized;
    float x_min = *std::min_element(x_data.begin(), x_data.end());
    float x_max = *std::max_element(x_data.begin(), x_data.end());
    x_data_normalized.reserve(x_data.size());
    std::vector<float> y_data_normalized;
    y_data_normalized.reserve(y_data.size());
    for (auto y : y_data) {
        y_data_normalized.push_back(y / 100.0f);
    }
    for (auto x : x_data) {
        x_data_normalized.push_back((x - x_min) / (x_max - x_min));
    }
    int degree = 5;
    std::vector<float> coefficients(degree + 1);
    for (auto& c : coefficients) c = (rand() % 1000 / 1000.0f - 0.5f) * 0.1f; // -0.05 to +0.05
    polynomial_regression_gd(x_data_normalized, y_data_normalized, coefficients);

    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            //datapoints



            
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_Once);  // 800x600 pixels, set once

            ImGui::Begin("Polynomial Regression");
            
            if (ImPlot::BeginPlot("Grades Prediction"))
            {
                ImPlot::PlotScatter("Actual Grades", x_data.data(), y_data.data(), (int)x_data.size());

                static std::vector<float> x_pred(100);
                static std::vector<float> y_pred(100);
                if (x_pred.size() != 100) {
                    x_pred.resize(100);
                    y_pred.resize(100);
                }
                float x_pred_end = x_max + 5.0f;
                float x_step = (x_pred_end - x_min) / 100.0f;
                for (int i = 0; i < 100; i++) {
                    float x_real = x_min + x_step * i; // actual x
                    float x_norm = (x_real - x_min) / (x_max - x_min);     // normalized
                    x_pred[i] = x_real;
                    y_pred[i] = predict(x_norm, coefficients) * 100.0f;
                }
                ImPlot::PlotLine("Prediction", x_pred.data(), y_pred.data(), 100);
                ImPlot::EndPlot();
            }
            ImGui::End();

            ImGui::Render();

            float clear_color[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);

            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            g_pSwapChain->Present(1, 0);

        }
        
    }
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClass(wc.lpszClassName, wc.hInstance);
    return (int)msg.wParam;
}
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
    {
        return true;
    }
    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

        // Handle other messages here if needed...

    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}