clear
cd G:\NSD_LOCAL_DATA
addpath(genpath(pwd));
cd NSD_fMRI\

load nsd_expdesign.mat
shared_1000_idx = sharedix;
interested_subject_all = [1,2,5,7];
interested_hemis = ['l','r'];

for interested_subject = interested_subject_all
    subject_order_here = subjectim(interested_subject, masterordering);
    for hh = 1:2
        tic
        fprintf('Process s% %h',interested_subject, hh)
        surf_data_file = fullfile('Dataset/',sprintf('S%d', interested_subject),'fs','surf',sprintf('flat_%sh.gii', interested_hemis(hh)));
        surf_data = gifti(surf_data_file);
        vertex_num = length(surf_data.vertices);
        all_brain_data = int16(zeros([vertex_num, 3000]));
        shared_location = zeros([1,3000]);
        location_idx = 1;

        for ses_idx = 1:40
            file_name = sprintf('%sh.betas_session%02d.hdf5',interested_hemis(hh), ses_idx);
            dpath_here = fullfile('Dataset',sprintf('S%d', interested_subject),'fsnative_beta','betas_fithrf_GLMdenoise_RR',file_name);
            data_readout = h5read(dpath_here,'/betas',[1,1], [vertex_num, 750]);
            order_idx = subject_order_here(1+(ses_idx-1)*750:ses_idx*750);
            for trials = 1:750
                if(find(order_idx(trials)==shared_1000_idx))
                    img_here = order_idx(trials);
                    all_brain_data(:,location_idx)=data_readout(:, trials);
                    if(mean(data_readout(:, trials))==0)
                        keyboard
                    end
                    shared_location(location_idx)=img_here;
                    location_idx = location_idx+1;
                end
            end
            fprintf('Loading data %d from %d ses %sh S%d \n',ses_idx,40,interested_hemis(hh),interested_subject)
        end
        mean_brain_data = int16(zeros([vertex_num, 1000]));
        
        data_trial_wise = zeros([vertex_num, 3, 1000]);
        r_pool = zeros([vertex_num,3]);
        for interested_nsd_img = 1:1000
            loc_here = find(shared_location==shared_1000_idx(interested_nsd_img));
            mean_brain_data(:,interested_nsd_img) = mean(all_brain_data(:, loc_here),2);
            data_trial_wise(:,:,interested_nsd_img) = all_brain_data(:, loc_here);
        end

        for leaved_trial = 1:3
            kept_trial = setdiff(1:3, leaved_trial);
            rsp1 = squeeze(data_trial_wise(:,leaved_trial,:));
            rsp2 = squeeze(mean(data_trial_wise(:, kept_trial, :),2));
            for vertex_now = 1:vertex_num
                r_pool(vertex_now, leaved_trial) = corr(rsp1(vertex_now,:)',rsp2(vertex_now,:)');
                disp(vertex_now)
            end
            % disp(leaved_trial)
        end
        r_pool = mean(r_pool');
        r_pool = 2*r_pool./(1+r_pool);
        save(fullfile('Dataset',sprintf('S%d_%sh_Rsp.mat',interested_subject,interested_hemis(hh))),'mean_brain_data','r_pool')
        fprintf('%.02f min', toc./60)
    end
end