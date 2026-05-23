from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
from django.core.serializers.json import DjangoJSONEncoder
import json


from .forms import RegisterForm, LoginForm, CustomAlgorithmForm, RenameAlgorithmForm, SimulationConfigForm
from .models import CustomAlgorithm, SimulationConfig


def register_view(request):
    if request.user.is_authenticated:
        return redirect('simulations:setup')
    
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('simulations:setup')
    else:
        form = RegisterForm()
    
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('simulations:setup')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                next_url = request.GET.get('next', 'simulations:setup')
                return redirect(next_url)
        messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('simulations:setup')


@login_required
def profile_view(request):
    algorithms = request.user.algorithms.filter(is_active=True)
    configs = request.user.configs.all()
    return render(request, 'accounts/profile.html', {
        'algorithms': algorithms,
        'configs': configs,
        'user': request.user,
    })


@login_required
def upload_algorithm(request):
    if request.method == 'POST':
        form = CustomAlgorithmForm(request.POST, request.FILES)
        if form.is_valid():
            algorithm = form.save(commit=False)
            algorithm.user = request.user
            algorithm.save()
            
            algorithm_class = algorithm.get_algorithm_class()
            if algorithm_class:
                messages.success(request, f'Algorithm "{algorithm.name}" uploaded and validated successfully!')
            else:
                messages.warning(request, f'Algorithm "{algorithm.name}" uploaded but could not find a TrackingAlgorithm subclass.')
            
            return redirect('accounts:profile')
    else:
        form = CustomAlgorithmForm()
    
    return render(request, 'accounts/upload_algorithm.html', {'form': form})


@login_required
def delete_algorithm(request, algorithm_id):
    algorithm = get_object_or_404(CustomAlgorithm, id=algorithm_id, user=request.user)
    algorithm_name = algorithm.name
    algorithm.delete()
    messages.success(request, f'Algorithm "{algorithm_name}" deleted successfully!')
    return redirect('accounts:profile')


@login_required
def rename_algorithm(request, algorithm_id):
    algorithm = get_object_or_404(CustomAlgorithm, id=algorithm_id, user=request.user)
    
    if request.method == 'POST':
        form = RenameAlgorithmForm(request.POST)
        if form.is_valid():
            new_name = form.cleaned_data['new_name']
            algorithm.name = new_name
            algorithm.save()
            messages.success(request, f'Algorithm renamed to "{new_name}"!')
            return redirect('accounts:profile')
    else:
        form = RenameAlgorithmForm(initial={'new_name': algorithm.name})
    
    return render(request, 'accounts/rename_algorithm.html', {
        'form': form,
        'algorithm': algorithm,
    })


@login_required
def save_config(request):
    is_russian = request.LANGUAGE_CODE == 'ru'
    
    if request.method == 'POST':
        form = SimulationConfigForm(request.POST, user=request.user)
        if form.is_valid():
            config = form.save(commit=False)
            config.user = request.user
            
            algorithms = request.POST.getlist('algorithms')
            config.algorithms = algorithms
            
            lline_config = {}
            for algo in algorithms:
                lline_config[algo] = request.POST.get(f'lline_{algo}') == 'on'
            config.lline_config = lline_config
            
            adjacency_matrix_str = request.POST.get('adjacency_matrix', '')
            if adjacency_matrix_str:
                try:
                    rows = adjacency_matrix_str.strip().split('\n')
                    adjacency_matrix = []
                    for row in rows:
                        adjacency_matrix.append([int(x.strip()) for x in row.split(',')])
                    if len(adjacency_matrix) == config.num_sensors:
                        valid = True
                        for row in adjacency_matrix:
                            if len(row) != config.num_sensors:
                                valid = False
                                break
                        if valid:
                            config.adjacency_matrix = adjacency_matrix
                            config.adjacency_sparsity = None
                except:
                    pass
            
            if config.adjacency_sparsity is not None and config.adjacency_sparsity < 100:
                config.adjacency_matrix = None
            
            config.save()
            
            if is_russian:
                messages.success(request, f'Конфигурация "{config.name}" успешно сохранена!')
            else:
                messages.success(request, f'Configuration "{config.name}" saved successfully!')
            
            return redirect('accounts:profile')
    else:
        initial_data = {}
        if 'params' in request.GET:
            try:
                params = json.loads(request.GET.get('params'))
                initial_data = {
                    'duration': params.get('duration', 50),
                    'num_sensors': params.get('num_sensors', 3),
                    'num_linear_targets': params.get('num_linear_targets', 2),
                    'num_random_targets': params.get('num_random_targets', 2),
                    'num_runs': params.get('num_runs', 1),
                    'algorithms': params.get('algorithms', []),
                    'noise_enabled': params.get('noise_enabled', False),
                    'noise_type': params.get('noise_type', 'uniform'),
                    'noise_low': params.get('noise_low', -0.1),
                    'noise_high': params.get('noise_high', 0.1),
                    'noise_mean': params.get('noise_mean', 0.0),
                    'noise_std': params.get('noise_std', 0.1),
                }
            except:
                pass
        
        form = SimulationConfigForm(initial=initial_data, user=request.user)
    
    available_algorithms = ['original_spsa', 'accelerated_spsa', 'distributed_kalman_filter']
    custom_algorithms = list(request.user.algorithms.filter(is_active=True).values_list('name', flat=True))
    all_algorithms = available_algorithms + custom_algorithms
    
    return render(request, 'accounts/save_config.html', {
        'form': form,
        'all_algorithms': all_algorithms,
    })


@login_required
def delete_config(request, config_id):
    config = get_object_or_404(SimulationConfig, id=config_id, user=request.user)
    config_name = config.name
    config.delete()
    
    is_russian = request.LANGUAGE_CODE == 'ru'
    if is_russian:
        messages.success(request, f'Конфигурация "{config_name}" успешно удалена!')
    else:
        messages.success(request, f'Configuration "{config_name}" deleted successfully!')
    
    return redirect('accounts:profile')


@login_required
def run_config(request, config_id):
    config = get_object_or_404(SimulationConfig, id=config_id, user=request.user)
    params = config.to_params_dict()
    request.session['simulation_params'] = params
    request.session['single_config_name'] = config.name
    return redirect('simulations:results')


@login_required
def run_multiple_configs(request):
    if request.method == 'POST':
        config_ids = request.POST.getlist('config_ids')
        if not config_ids:
            is_russian = request.LANGUAGE_CODE == 'ru'
            if is_russian:
                messages.error(request, 'Пожалуйста, выберите хотя бы одну конфигурацию для запуска.')
            else:
                messages.error(request, 'Please select at least one configuration to run.')
            return redirect('accounts:profile')
        
        configs = SimulationConfig.objects.filter(id__in=config_ids, user=request.user)
        if not configs:
            is_russian = request.LANGUAGE_CODE == 'ru'
            if is_russian:
                messages.error(request, 'Не выбрано ни одной действительной конфигурации.')
            else:
                messages.error(request, 'No valid configurations selected.')
            return redirect('accounts:profile')
        
        all_params = []
        for config in configs:
            params = config.to_params_dict()
            all_params.append({
                'name': config.name,
                'params': params
            })
        
        request.session['multiple_configs'] = all_params
        return redirect('simulations:comparison_results')
    
    return redirect('accounts:profile')